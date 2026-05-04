import copy
import json
import os
import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval


VIRCONV_PROFILE_STAGES = (
    'Pre-processing & RSV Warp',
    'Voxelization',
    'VFE',
    '3D Backbone',
    'BEV Compression & 2D Backbone',
    'Dense Head',
    'ROI Head',
    'Post-processing & RSV Inverse',
)


def _batch_value_to_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return _batch_value_to_bool(value[0], default=default)
    return bool(value)


def _measure_cuda_stage(stage_fn, device):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device=device)
    start_event.record()
    stage_output = stage_fn()
    end_event.record()
    torch.cuda.synchronize(device=device)

    return stage_output, start_event.elapsed_time(end_event)


def _validate_nonempty_dataloader(cfg, dataloader):
    if len(dataloader) == 0:
        raise ValueError(
            'The evaluation dataloader is empty. '
            f'Check DATA_CONFIG.DATA_PATH={cfg.DATA_CONFIG.DATA_PATH} and ensure KITTI infos/split files exist.'
        )


class VirConvLatencyProfiler:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model.module if hasattr(model, 'module') else model
        self.device = next(self.model.parameters()).device
        self.use_uvw_coords = bool(getattr(self.model, 'use_uvw_coords', False))
        self.voxel_cfg = self._get_voxel_processor_cfg()
        self.voxelizers = {}

    def _get_voxel_processor_cfg(self):
        for processor_cfg in self.cfg.DATA_CONFIG.DATA_PROCESSOR:
            if processor_cfg.NAME in ['transform_points_to_voxels', 'transform_points_to_radial_voxels']:
                return processor_cfg
        raise ValueError('VirConv latency profiling requires a voxelization processor in cfg.DATA_CONFIG.DATA_PROCESSOR')

    def _get_coors_range(self):
        coors_range = self.model.dataset.uvw_range if self.use_uvw_coords else self.model.dataset.point_cloud_range
        if coors_range is None:
            raise ValueError('Voxelization range is missing from the dataset')
        if isinstance(coors_range, np.ndarray):
            return coors_range.tolist()
        return list(coors_range)

    def _get_voxelizer(self, num_point_features):
        from spconv.pytorch.utils import PointToVoxel

        num_point_features = int(num_point_features)
        voxelizer = self.voxelizers.get(num_point_features, None)
        if voxelizer is None:
            voxelizer = PointToVoxel(
                vsize_xyz=[float(x) for x in self.voxel_cfg.VOXEL_SIZE],
                coors_range_xyz=[float(x) for x in self._get_coors_range()],
                num_point_features=num_point_features,
                max_num_voxels=int(self.voxel_cfg.MAX_NUMBER_OF_VOXELS['test']),
                max_num_points_per_voxel=int(self.voxel_cfg.MAX_POINTS_PER_VOXEL),
                device=self.device
            )
            self.voxelizers[num_point_features] = voxelizer
        return voxelizer

    def _extract_single_batch_points(self, batch_dict, key='points'):
        batch_size = int(batch_dict.get('batch_size', 0))
        if batch_size != 1:
            raise ValueError(f'VirConv latency profiling only supports batch_size=1, got {batch_size}')

        points = batch_dict.get(key, None)
        if not torch.is_tensor(points):
            raise TypeError(f'batch_dict["{key}"] must be a torch.Tensor after load_data_to_gpu')
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError(f'Unexpected {key} tensor shape: {tuple(points.shape)}')

        batch_indices = points[:, 0]
        if batch_indices.numel() > 0 and not torch.all(batch_indices == 0):
            raise ValueError(f'VirConv latency profiling expects all {key} batch indices to be zero for batch_size=1')

        point_features = points[:, 1:].contiguous()
        if point_features.shape[0] == 0:
            raise ValueError(f'Encountered an empty point cloud in "{key}" while profiling latency')
        return point_features

    def _reorder_lidar_first(self, point_features):
        if not self.voxel_cfg.get('LIDAR_FIRST', False) or point_features.shape[1] == 0:
            return point_features

        point_type = point_features[:, -1]
        points_lidar = point_features[point_type == 2]
        points_virtual = point_features[point_type == 1]
        if points_lidar.shape[0] + points_virtual.shape[0] != point_features.shape[0]:
            return point_features
        return torch.cat((points_lidar, points_virtual), dim=0).contiguous()

    def _warp_point_features(self, point_features):
        if not self.use_uvw_coords:
            return point_features

        xyz = point_features[:, :3]
        point_tail_features = point_features[:, 3:]
        warped_xyz = self.model.rsv_scaler.warp_points(xyz)
        if point_tail_features.shape[1] == 0:
            return warped_xyz.contiguous()
        return torch.cat((warped_xyz, point_tail_features), dim=1).contiguous()

    def preprocess_points(self, batch_dict):
        processed_points = {}
        point_features = self._extract_single_batch_points(batch_dict, key='points')
        point_features = self._reorder_lidar_first(point_features)
        processed_points['points'] = self._warp_point_features(point_features)

        if torch.is_tensor(batch_dict.get('points_mm', None)):
            processed_points['points_mm'] = self._warp_point_features(
                self._extract_single_batch_points(batch_dict, key='points_mm')
            )

        return processed_points

    def voxelize_points(self, processed_points, batch_dict):
        use_lead_xyz = _batch_value_to_bool(batch_dict.get('use_lead_xyz', True), default=True)
        branch_suffix = {
            'points': '',
            'points_mm': '_mm',
        }

        for point_key, points in processed_points.items():
            voxelizer = self._get_voxelizer(points.shape[1])
            voxels, voxel_coords, voxel_num_points = voxelizer(points)

            if not use_lead_xyz:
                voxels = voxels[..., 3:]

            batch_column = torch.zeros((voxel_coords.shape[0], 1), dtype=voxel_coords.dtype, device=voxel_coords.device)
            suffix = branch_suffix[point_key]
            batch_dict['voxels' + suffix] = voxels.contiguous()
            batch_dict['voxel_coords' + suffix] = torch.cat((batch_column, voxel_coords), dim=1).contiguous()
            batch_dict['voxel_num_points' + suffix] = voxel_num_points

        return batch_dict

    def run_vfe(self, batch_dict):
        if getattr(self.model, 'vfe', None) is None:
            raise ValueError('VirConv latency profiling expected model.vfe to be present')
        return self.model.vfe(batch_dict)

    def run_backbone_3d(self, batch_dict):
        if getattr(self.model, 'backbone_3d', None) is None:
            raise ValueError('VirConv latency profiling expected model.backbone_3d to be present')
        return self.model.backbone_3d(batch_dict)

    def run_bev_backbone(self, batch_dict):
        if getattr(self.model, 'map_to_bev_module', None) is not None:
            batch_dict = self.model.map_to_bev_module(batch_dict)
        if getattr(self.model, 'backbone_2d', None) is not None:
            batch_dict = self.model.backbone_2d(batch_dict)
        return batch_dict

    def run_dense_head(self, batch_dict):
        if getattr(self.model, 'dense_head', None) is None:
            raise ValueError('VirConv latency profiling expected model.dense_head to be present')
        return self.model.dense_head(batch_dict)

    def run_roi_head(self, batch_dict):
        if getattr(self.model, 'roi_head', None) is None:
            return batch_dict
        return self.model.roi_head(batch_dict)

    def run_post_processing(self, batch_dict):
        batch_dict['infer_time'] = True
        return self.model.post_processing(batch_dict)

    def profile_batch(self, batch_dict):
        stage_times_ms = {}
        total_start_event = torch.cuda.Event(enable_timing=True)
        total_end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device=self.device)
        total_start_event.record()

        processed_points, stage_times_ms[VIRCONV_PROFILE_STAGES[0]] = _measure_cuda_stage(
            lambda: self.preprocess_points(batch_dict), self.device
        )
        batch_dict, stage_times_ms[VIRCONV_PROFILE_STAGES[1]] = _measure_cuda_stage(
            lambda: self.voxelize_points(processed_points, batch_dict), self.device
        )
        batch_dict, stage_times_ms[VIRCONV_PROFILE_STAGES[2]] = _measure_cuda_stage(
            lambda: self.run_vfe(batch_dict), self.device
        )
        batch_dict, stage_times_ms[VIRCONV_PROFILE_STAGES[3]] = _measure_cuda_stage(
            lambda: self.run_backbone_3d(batch_dict), self.device
        )
        batch_dict, stage_times_ms[VIRCONV_PROFILE_STAGES[4]] = _measure_cuda_stage(
            lambda: self.run_bev_backbone(batch_dict), self.device
        )
        batch_dict, stage_times_ms[VIRCONV_PROFILE_STAGES[5]] = _measure_cuda_stage(
            lambda: self.run_dense_head(batch_dict), self.device
        )
        batch_dict, stage_times_ms[VIRCONV_PROFILE_STAGES[6]] = _measure_cuda_stage(
            lambda: self.run_roi_head(batch_dict), self.device
        )
        _, stage_times_ms[VIRCONV_PROFILE_STAGES[7]] = _measure_cuda_stage(
            lambda: self.run_post_processing(batch_dict), self.device
        )

        total_end_event.record()
        torch.cuda.synchronize(device=self.device)
        total_time_ms = total_start_event.elapsed_time(total_end_event)
        return stage_times_ms, total_time_ms


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=True, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        #begin = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)
        disp_dict = {}
        #end = time.time()
        #print(end-begin)

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    path = result_dir / 'result.pkl'
    if os.path.exists(path):
        path = result_dir / ('result_'+str(time.time())[:10]+'.pkl')

    with open(path, 'wb') as f:
        pickle.dump(det_annos, f)
    
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    
    return ret_dict

def eval_one_epoch_dist(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=True, result_dir=None):
    
    # ---------------- helpers: distance-binned filtering ----------------
    RANGE_BINS = [(0.0, 30.0), (30.0, 50.0), (50.0, 70.0)]      # distance bins in meters

    def _compute_center_dist_lidar(boxes_lidar):
        """
        Compute 2D radial distance in LiDAR coordinates for boxes。
        boxes_lidar: [N,7 or 9], centers at [:,0:2] in LiDAR (x,y)
        distance is sqrt(x^2 + y^2)
        """
        if boxes_lidar is None or len(boxes_lidar) == 0:        # handle empty input
            return np.zeros((0,), dtype=np.float32)
        xy = boxes_lidar[:, :2]                                 # take x, y
        return np.sqrt((xy ** 2).sum(axis=1))                   # Euclidean radius in x-y plane

    def _filter_det_annos_by_range(det_annos_in, rmin, rmax):
        """
        Filter prediction dicts by [rmin, rmax) using LiDAR-plane distance.
        Only fields whose length == num_objects will be masked; others kept.
        """
        det_out = []                                            # output list of filtered annos
        for anno in det_annos_in:
            out = {}
            num = len(anno.get('name', []))

            for k, v in anno.items():
                out[k] = v

            if ('boxes_lidar' in anno) and num > 0:
                d = _compute_center_dist_lidar(anno['boxes_lidar'])
                d = np.asarray(d)
                mask = (d >= rmin) & (d < rmax)

                idx = np.where(mask)[0]

                for k, v in anno.items():

                    if k != 'frame_id':
                        vv = np.asarray(v)
                        if idx.size == 0:
                            slicer = (slice(0, 0),) + (slice(None),) * (vv.ndim - 1)
                            out[k] = vv[slicer]
                        else:
                            out[k] = vv[idx]
                    else:
                        out[k] = v

            det_out.append(out)
        return det_out

    def _filter_gt_annos_by_range(gt_annos_in, rmin, rmax):
        """
        Filter ground-truth dicts by [rmin, rmax) using LiDAR-plane distance.
        Only fields whose length == num_objects will be masked; others kept.
        """
        gt_out = []                                            # output list of filtered annos
        for anno in gt_annos_in:
            out = {}
            num = len(anno.get('name', []))

            for k, v in anno.items():
                out[k] = v

            if ('gt_boxes_lidar' in anno) and num > 0:
                d = _compute_center_dist_lidar(anno['gt_boxes_lidar'])
                d = np.asarray(d)
                mask = (d >= rmin) & (d < rmax)

                idx = np.where(mask)[0]

                for k, v in anno.items():

                    if k != 'frame_id':
                        vv = np.asarray(v)
                        if idx.size == 0:
                            slicer = (slice(0, 0),) + (slice(None),) * (vv.ndim - 1)
                            out[k] = vv[slicer]
                        else:
                            out[k] = vv[idx]
                    else:
                        out[k] = v

            gt_out.append(out)
        return gt_out
    
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        #begin = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)
        disp_dict = {}
        #end = time.time()
        #print(end-begin)

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    path = result_dir / 'result.pkl'
    if os.path.exists(path):
        path = result_dir / ('result_'+str(time.time())[:10]+'.pkl')

    with open(path, 'wb') as f:
        pickle.dump(det_annos, f)
    
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # -------- distance-binned evaluations (prediction-filtered) --------
    if 'annos' not in dataset.kitti_infos[0]:
        logger.info("No ground truth annotations found (Test Set). Skipping evaluation.")
        return ret_dict
    # ground-truth annos
    gt_annos = [copy.deepcopy(info['annos']) for info in dataset.kitti_infos]

    # -------------------- overall evaluations --------------------------
    result_str_overall, result_dict_overall = kitti_eval.get_official_eval_result(
        gt_annos, det_annos, class_names)
    logger.info(f'===== Overall Results =====')
    logger.info(result_str_overall)

    for (rmin, rmax) in RANGE_BINS:
        det_annos_bin = _filter_det_annos_by_range(det_annos, rmin, rmax)
        gt_annos_bin = _filter_gt_annos_by_range(gt_annos, rmin, rmax)
        try:
            result_str_bin, result_dict_bin = kitti_eval.get_official_eval_result(
                gt_annos_bin, det_annos_bin, class_names)

            logger.info(f'===== Range [{rmin:.0f},{rmax:.0f}) m =====')
            logger.info(result_str_bin)
            # prefix keys for clarity, e.g., 'mAP_R0_30'
            for k, v in result_dict_bin.items():
                ret_dict[f'{k}_R{int(rmin)}_{int(rmax)}'] = v
        except TypeError:
            logger.warning(f'evaluation() signature differs on this dataset; please adapt if needed.')

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    
    return ret_dict

def cal_inference_time(cfg, model, dataloader, logger):
    logger.info('*************** CALCULATING INFERENCE TIME *****************')
    
    model.eval()
    
    infer_time_meter = common_utils.AverageMeter()
    
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='Inference Speed', dynamic_ncols=True)
    
    # warmup iterations
    warmup_iter = 88
    
    for i, batch_dict in enumerate(dataloader):
        batch_dict['infer_time'] = True
        
        load_data_to_gpu(batch_dict)

        # ---------------- Warmup phase ----------------
        if i < warmup_iter:
            with torch.no_grad():
                model(batch_dict)
            
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix({'status': f'Warmup {i+1}/{warmup_iter}'})
                progress_bar.update()
            continue

        # ---------------- Actual inference timing phase ----------------
        
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            # PCDet 的 model() 在 eval 模式下通常包含了 NMS 等后处理
            pred_dicts, ret_dict, _ = model(batch_dict)

        torch.cuda.synchronize()
        inference_time = time.time() - start_time

        # Convert to milliseconds (ms)
        infer_time_meter.update(inference_time * 1000)
        
        if cfg.LOCAL_RANK == 0:
            disp_dict = {
                    'latency': f'{infer_time_meter.val:.2f}ms ({infer_time_meter.avg:.2f}ms)',
                    'status': 'Testing'
                }
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # Output final report
    avg_latency = infer_time_meter.avg
    fps = 1000.0 / avg_latency
    
    logger.info('**************** Inference Time Results *****************')
    logger.info(f'cfg.tag       : {cfg.TAG}')
    logger.info(f'Total Samples: {len(dataloader.dataset)}')
    logger.info(f'Warmup Iters : {warmup_iter}')
    logger.info(f'Avg Latency  : {avg_latency:.2f} ms / frame')
    logger.info(f'FPS          : {fps:.2f} frame / s')
    logger.info('*********************************************************')

    return avg_latency, fps


def profile_latency_breakdown(cfg, args, model, dataloader, logger, result_dir=None):
    logger.info('*************** PROFILING VIRCONV LATENCY BREAKDOWN *****************')

    model.eval()
    profiler = VirConvLatencyProfiler(cfg, model)

    _validate_nonempty_dataloader(cfg, dataloader)

    warmup_iters = max(int(getattr(args, 'profile_warmup_iters', 20)), 0)
    requested_profile_iters = max(int(getattr(args, 'profile_max_iters', 0)), 0)
    available_profile_iters = len(dataloader) - warmup_iters
    if available_profile_iters <= 0:
        raise ValueError(
            f'Warm-up iters ({warmup_iters}) must be smaller than the dataloader length ({len(dataloader)})'
        )

    measured_target = available_profile_iters if requested_profile_iters == 0 else min(requested_profile_iters, available_profile_iters)
    stage_meters = {stage_name: common_utils.AverageMeter() for stage_name in VIRCONV_PROFILE_STAGES}
    total_meter = common_utils.AverageMeter()

    progress_total = warmup_iters + measured_target
    progress_bar = None
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=progress_total, leave=True, desc='latency_profile', dynamic_ncols=True)

    measured_iters = 0
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            if measured_iters >= measured_target:
                break

            load_data_to_gpu(batch_dict)

            if batch_idx < warmup_iters:
                profiler.profile_batch(batch_dict)
                if progress_bar is not None:
                    progress_bar.set_postfix({'phase': f'warmup {batch_idx + 1}/{warmup_iters}'})
                    progress_bar.update()
                continue

            stage_times_ms, total_time_ms = profiler.profile_batch(batch_dict)
            for stage_name, stage_time_ms in stage_times_ms.items():
                stage_meters[stage_name].update(stage_time_ms)
            total_meter.update(total_time_ms)
            measured_iters += 1

            if progress_bar is not None:
                progress_bar.set_postfix({
                    'phase': 'measure',
                    'total_ms': f'{total_meter.val:.2f} ({total_meter.avg:.2f})'
                })
                progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()

    if measured_iters == 0:
        raise ValueError('No iterations were profiled after warm-up')

    total_avg_ms = float(total_meter.avg)
    stage_sum_avg_ms = float(sum(stage_meters[stage_name].avg for stage_name in VIRCONV_PROFILE_STAGES))

    logger.info('**************** VirConv Latency Breakdown *****************')
    logger.info(f'Warmup Iters   : {warmup_iters}')
    logger.info(f'Measured Iters : {measured_iters}')
    logger.info(f'Total Avg      : {total_avg_ms:.3f} ms / frame')
    logger.info(f'Stage Sum Avg  : {stage_sum_avg_ms:.3f} ms / frame')
    logger.info(f'Gap            : {abs(total_avg_ms - stage_sum_avg_ms):.3f} ms')
    logger.info(f'{"Stage":<40} {"Avg(ms)":>12} {"PctTotal":>10} {"Iters":>8}')
    for stage_name in VIRCONV_PROFILE_STAGES:
        avg_ms = float(stage_meters[stage_name].avg)
        pct_total = (avg_ms / total_avg_ms * 100.0) if total_avg_ms > 0 else 0.0
        logger.info(f'{stage_name:<40} {avg_ms:>12.3f} {pct_total:>9.2f}% {stage_meters[stage_name].count:>8}')
    logger.info('***********************************************************')

    result_dict = {
        'model_name': cfg.MODEL.NAME,
        'cfg_file': args.cfg_file,
        'warmup_iters': warmup_iters,
        'measured_iters': measured_iters,
        'stage_stats': {},
        'total_avg_ms': total_avg_ms,
        'stage_sum_avg_ms': stage_sum_avg_ms,
    }
    for stage_name in VIRCONV_PROFILE_STAGES:
        avg_ms = float(stage_meters[stage_name].avg)
        result_dict['stage_stats'][stage_name] = {
            'avg_ms': avg_ms,
            'pct_total': (avg_ms / total_avg_ms * 100.0) if total_avg_ms > 0 else 0.0,
            'num_iters': int(stage_meters[stage_name].count),
        }

    if getattr(args, 'profile_save_json', False) and result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        output_json = result_dir / 'latency_breakdown.json'
        with open(output_json, 'w') as f:
            json.dump(result_dict, f, indent=2)
        logger.info('Latency breakdown JSON is saved to %s' % output_json)

    return result_dict

if __name__ == '__main__':
    pass
