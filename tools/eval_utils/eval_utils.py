import pickle
import time

import numpy as np
import torch
import tqdm
import time
import copy
import os

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval


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

if __name__ == '__main__':
    pass
