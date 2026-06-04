import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None, rot_num=1):
        super().__init__()
        self.rot_num = rot_num
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range
        self.use_uvw_coords = config.get('USE_UVW_COORDS', False)
        if self.use_uvw_coords:
            self.r_far = config.R_FAR
            self.s_max = config.S_MAX
            self.beta = config.BETA
            self.schedule = self._normalize_schedule(config.get('SCHEDULE', 'exp'))

    @staticmethod
    def _normalize_schedule(schedule):
        schedule = str(schedule).lower()
        if schedule in ('exp', 'exponential'):
            return 'exp'
        if schedule in ('linear', 'lin'):
            return 'linear'
        raise ValueError(f"Unsupported RSV schedule: {schedule}")

    def _scale_of_dist(self, dist):
        if self.schedule == 'linear':
            return np.where(
                dist < self.r_far,
                self.s_max - (self.s_max - 1.0) * dist / self.r_far,
                1.0
            )
        return 1 + (self.s_max - 1) * np.exp(-self.beta * dist / self.r_far)

    @staticmethod
    def _normalize_log_scale(log_s):
        log_s_range = log_s.max() - log_s.min()
        if log_s_range <= 1e-8:
            return np.zeros_like(log_s)
        return (log_s - log_s.min()) / log_s_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """

        for i in range(self.rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            data_dict['points'+rot_num_id], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                data_dict['points'+rot_num_id]
            )

            if 'mm' in data_dict:
                data_dict['points_mm'+rot_num_id], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                    data_dict['points_mm'+rot_num_id]
                )

        data_dict['use_lead_xyz'] = use_lead_xyz

        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        assert points.shape[-1] == len(self.src_feature_list)

        if self.use_uvw_coords:
            x = points[:, 0]
            y = points[:, 1]

            r_max_pc = np.sqrt(
                (self.point_cloud_range[3] - self.point_cloud_range[0]) ** 2 +
                (self.point_cloud_range[4] - self.point_cloud_range[1]) ** 2
            )
            dist = np.sqrt(x ** 2 + y ** 2)
            dist_norm = np.clip(dist / (r_max_pc + 1e-6), 0.0, 1.0)
            log_dist = np.log1p(dist)
            log_dist_norm = log_dist / np.log1p(r_max_pc)

            s = self._scale_of_dist(dist)
            log_s = np.log(s)
            log_s_norm = self._normalize_log_scale(log_s)

            point_feature_list = [points[:, 0:3]]
            for key in self.used_feature_list:
                if key in ['x', 'y', 'z']:
                    continue

                if key == 'intensity':
                    idx = self.src_feature_list.index('intensity')
                    point_feature_list.append(points[:, idx:idx + 1])
                elif key == 'timestamp':
                    idx = self.src_feature_list.index('timestamp')
                    point_feature_list.append(points[:, idx:idx + 1])
                elif key == 'dist':
                    point_feature_list.append(dist_norm[:, None])
                elif key == 'log_dist':
                    point_feature_list.append(log_dist_norm[:, None])
                elif key == 'scale':
                    point_feature_list.append(log_s_norm[:, None])
                else:
                    raise NotImplementedError(f"Unknown feature: {key}")
        else:
            point_feature_list = [points[:, 0:3]]
            for key in self.used_feature_list:
                if key in ['x', 'y', 'z']:
                    continue
                idx = self.src_feature_list.index(key)
                point_feature_list.append(points[:, idx:idx + 1])

        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True

    def absolute_coordinates_encoding_mm(self, points=None):
        if points is None:
            num_output_features = self.point_encoding_config.num_features
            return num_output_features

        if self.use_uvw_coords:
            ori_num_output_features = self.point_encoding_config.num_features - 3
            point_feature_list = [points[:, 0:ori_num_output_features - 1]]
            x = points[:, 0]
            y = points[:, 1]

            r_max_pc = np.sqrt((self.point_cloud_range[3] - self.point_cloud_range[0])**2 + (self.point_cloud_range[4] - self.point_cloud_range[1])**2)
            dist = np.sqrt(x**2 + y**2)
            dist_norm = np.clip(dist / (r_max_pc + 1e-6), 0.0, 1.0)
            log_dist = np.log1p(dist)
            log_dist_norm = log_dist / np.log1p(r_max_pc)
            point_feature_list.append(dist_norm[:, None])
            point_feature_list.append(log_dist_norm[:, None])

            s = self._scale_of_dist(dist)
            log_s = np.log(s)
            log_s_norm = self._normalize_log_scale(log_s)
            point_feature_list.append(log_s_norm[:, None])
            point_feature_list.append(points[:, -1:])
            point_features = np.concatenate(point_feature_list, axis=1)
        else:
            point_features = points[:, 0:self.point_encoding_config.num_features]

        return point_features, True
