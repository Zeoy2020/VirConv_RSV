# pcdet/models/box_adapters/rsv_box_adapter.py
import numpy as np
import torch


class RSVBoxAdapter:
    """
    Adapter that warps 3D boxes between Cartesian (xyz) and radial-scaled (uvw) spaces.
    - `center_scale`: legacy approximation that uses the center scale s(r_c).
    - `four_corners_fixed_yaw`: warp the 4 BEV corners independently, then fit the
      smallest rectangle under the original yaw. This is still a rectangle proxy,
      but it is much closer to the true warped quadrilateral.

    Supports both torch.Tensor and numpy.ndarray inputs.
    """
    CENTER_SCALE_METHODS = {'center', 'center_scale'}
    FOUR_CORNERS_FIXED_YAW_METHODS = {
        'four_corners_fixed_yaw',
        'fixed_yaw_corners',
        '4corners_fixed_yaw',
    }

    def __init__(self, scaler, box_warp_method='center_scale'):
        self.scaler = scaler
        self.box_warp_method = str(box_warp_method).lower()
        if self.box_warp_method not in self.CENTER_SCALE_METHODS | self.FOUR_CORNERS_FIXED_YAW_METHODS:
            raise ValueError(f'Unsupported box_warp_method: {box_warp_method}')

    @staticmethod
    def _prepare_boxes(boxes):
        is_numpy = isinstance(boxes, np.ndarray)
        orig_shape = boxes.shape
        if is_numpy:
            tensor_boxes = torch.from_numpy(boxes).float()
            orig_dtype = None
        else:
            tensor_boxes = boxes.clone().to(dtype=torch.float32)
            orig_dtype = boxes.dtype
        flat_boxes = tensor_boxes.reshape(-1, tensor_boxes.shape[-1])
        return flat_boxes, is_numpy, orig_shape, orig_dtype

    @staticmethod
    def _restore_boxes(boxes, is_numpy, orig_shape, orig_dtype):
        boxes = boxes.reshape(orig_shape)
        if is_numpy:
            return boxes.cpu().numpy()
        return boxes.to(dtype=orig_dtype)

    @staticmethod
    def _boxes_to_bev_corners(boxes):
        num_boxes = boxes.shape[0]
        if num_boxes == 0:
            return boxes.new_zeros((0, 4, 2))

        template = boxes.new_tensor([
            [0.5, 0.5],
            [0.5, -0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
        ])
        half_sizes = boxes[:, None, 3:5] * template[None, :, :]
        yaw = boxes[:, 6]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        rot_mat = boxes.new_zeros((num_boxes, 2, 2))
        rot_mat[:, 0, 0] = cos_yaw
        rot_mat[:, 0, 1] = sin_yaw
        rot_mat[:, 1, 0] = -sin_yaw
        rot_mat[:, 1, 1] = cos_yaw

        corners = torch.matmul(half_sizes, rot_mat)
        corners += boxes[:, None, 0:2]
        return corners

    @staticmethod
    def _fit_fixed_yaw_rect(corners_xy, yaw):
        num_boxes = corners_xy.shape[0]
        if num_boxes == 0:
            return corners_xy.new_zeros((0, 4))

        cos_yaw = torch.cos(yaw).unsqueeze(-1)
        sin_yaw = torch.sin(yaw).unsqueeze(-1)
        local_x = corners_xy[..., 0] * cos_yaw + corners_xy[..., 1] * sin_yaw
        local_y = -corners_xy[..., 0] * sin_yaw + corners_xy[..., 1] * cos_yaw

        min_x = local_x.min(dim=1).values
        max_x = local_x.max(dim=1).values
        min_y = local_y.min(dim=1).values
        max_y = local_y.max(dim=1).values

        center_x_local = 0.5 * (min_x + max_x)
        center_y_local = 0.5 * (min_y + max_y)
        center_x = center_x_local * torch.cos(yaw) - center_y_local * torch.sin(yaw)
        center_y = center_x_local * torch.sin(yaw) + center_y_local * torch.cos(yaw)
        size_x = torch.clamp(max_x - min_x, min=1e-5)
        size_y = torch.clamp(max_y - min_y, min=1e-5)
        return torch.stack([center_x, center_y, size_x, size_y], dim=-1)

    def _warp_points_xy(self, points_xy):
        if points_xy.numel() == 0:
            return points_xy
        flat_points = points_xy.reshape(-1, 2)
        flat_points_3d = torch.cat([flat_points, flat_points.new_zeros((flat_points.shape[0], 1))], dim=-1)
        warped = self.scaler.warp_points(flat_points_3d)
        return warped[:, 0:2].reshape_as(points_xy)

    def _inv_warp_points_xy(self, points_xy):
        if points_xy.numel() == 0:
            return points_xy
        flat_points = points_xy.reshape(-1, 2)
        flat_points_3d = torch.cat([flat_points, flat_points.new_zeros((flat_points.shape[0], 1))], dim=-1)
        xyz = self.scaler.inv_warp_points(flat_points_3d)
        return xyz[:, 0:2].reshape_as(points_xy)

    def _warp_boxes_center_scale(self, boxes):
        x, y = boxes[:, 0], boxes[:, 1]
        eps = getattr(self.scaler, "eps", 1e-6)
        r = torch.sqrt(x * x + y * y + eps)
        s = self.scaler.s_of_r(r)

        warped = boxes.clone()
        warped[:, 0] = x * s
        warped[:, 1] = y * s
        warped[:, 3] = boxes[:, 3] * s
        warped[:, 4] = boxes[:, 4] * s
        return warped

    def _inv_warp_boxes_center_scale(self, boxes):
        warped = boxes.clone()
        points_uvw = warped[:, 0:3]
        xyz = self.scaler.inv_warp_points(points_uvw)
        warped[:, 0:3] = xyz

        x, y = xyz[:, 0], xyz[:, 1]
        eps = getattr(self.scaler, "eps", 1e-6)
        r = torch.sqrt(x * x + y * y + eps)
        s = torch.clamp(self.scaler.s_of_r(r), min=1e-6)
        warped[:, 3] = boxes[:, 3] / s
        warped[:, 4] = boxes[:, 4] / s
        return warped

    def _warp_boxes_four_corners_fixed_yaw(self, boxes):
        warped = boxes.clone()
        corners_xy = self._boxes_to_bev_corners(boxes)
        warped_corners_xy = self._warp_points_xy(corners_xy)
        fitted = self._fit_fixed_yaw_rect(warped_corners_xy, boxes[:, 6])
        warped[:, 0] = fitted[:, 0]
        warped[:, 1] = fitted[:, 1]
        warped[:, 3] = fitted[:, 2]
        warped[:, 4] = fitted[:, 3]
        return warped

    def _inv_warp_boxes_face_center(self, boxes):
        """
        High-quality closed-form initializer based on inverse-warping the 4 face
        centers of the UV proxy box.

        This avoids the strong inflation introduced by inverse-warping corners and
        refitting another rectangle in xyz space.
        """
        restored = boxes.clone()

        u = boxes[:, 0]
        v = boxes[:, 1]
        du = boxes[:, 3]
        dv = boxes[:, 4]
        yaw = boxes[:, 6]

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        half_du = du * 0.5
        half_dv = dv * 0.5

        # Front / back face centers along the local x-axis.
        uv_f_x = u + half_du * cos_yaw
        uv_f_y = v + half_du * sin_yaw
        uv_b_x = u - half_du * cos_yaw
        uv_b_y = v - half_du * sin_yaw

        # Left / right face centers along the local y-axis.
        uv_l_x = u - half_dv * sin_yaw
        uv_l_y = v + half_dv * cos_yaw
        uv_r_x = u + half_dv * sin_yaw
        uv_r_y = v - half_dv * cos_yaw

        face_centers_uv = torch.stack([
            torch.stack([uv_f_x, uv_f_y], dim=-1),
            torch.stack([uv_b_x, uv_b_y], dim=-1),
            torch.stack([uv_l_x, uv_l_y], dim=-1),
            torch.stack([uv_r_x, uv_r_y], dim=-1),
        ], dim=1)

        face_centers_xyz = self._inv_warp_points_xy(face_centers_uv)
        xyz_f = face_centers_xyz[:, 0, :]
        xyz_b = face_centers_xyz[:, 1, :]
        xyz_l = face_centers_xyz[:, 2, :]
        xyz_r = face_centers_xyz[:, 3, :]

        restored_dx = torch.norm(xyz_f - xyz_b, dim=-1)
        restored_dy = torch.norm(xyz_l - xyz_r, dim=-1)
        restored_x = face_centers_xyz[:, :, 0].mean(dim=1)
        restored_y = face_centers_xyz[:, :, 1].mean(dim=1)

        restored[:, 0] = restored_x
        restored[:, 1] = restored_y
        restored[:, 3] = restored_dx
        restored[:, 4] = restored_dy
        return restored

    def _inv_warp_boxes_iterative_root_finding(self, boxes_uv, max_iters=4, tol=1e-4):
        """
        Iteratively refine an xyz box so that its forward 4-corner proxy matches the
        predicted UV box as closely as possible.

        We start from the face-center inverse, then run a few pseudo-Newton updates
        using the local radial scale as a diagonal Jacobian approximation.
        """
        if boxes_uv.numel() == 0:
            return boxes_uv

        boxes_xyz = self._inv_warp_boxes_face_center(boxes_uv)
        target = boxes_uv[:, [0, 1, 3, 4]]
        eps = getattr(self.scaler, "eps", 1e-6)

        for _ in range(max(max_iters, 1)):
            current_uv_proxy = self._warp_boxes_four_corners_fixed_yaw(boxes_xyz)
            err = target - current_uv_proxy[:, [0, 1, 3, 4]]

            max_err = err.abs().sum(dim=1).max()
            if max_err < tol:
                break

            r = torch.sqrt(boxes_xyz[:, 0] ** 2 + boxes_xyz[:, 1] ** 2 + eps)
            s = torch.clamp(self.scaler.s_of_r(r), min=1e-6)

            boxes_xyz[:, 0] += err[:, 0] / s
            boxes_xyz[:, 1] += err[:, 1] / s
            boxes_xyz[:, 3] += err[:, 2] / s
            boxes_xyz[:, 4] += err[:, 3] / s
            boxes_xyz[:, 3:5] = torch.clamp(boxes_xyz[:, 3:5], min=1e-4)

        return boxes_xyz

    def _inv_warp_boxes_four_corners_fixed_yaw(self, boxes):
        return self._inv_warp_boxes_iterative_root_finding(boxes)

    def _warp_impl(self, boxes):
        if self.box_warp_method in self.CENTER_SCALE_METHODS:
            return self._warp_boxes_center_scale(boxes)
        return self._warp_boxes_four_corners_fixed_yaw(boxes)

    def _inv_warp_impl(self, boxes):
        if self.box_warp_method in self.CENTER_SCALE_METHODS:
            return self._inv_warp_boxes_center_scale(boxes)
        return self._inv_warp_boxes_four_corners_fixed_yaw(boxes)

    def warp_boxes(self, boxes_xyzwlh_yaw):
        """
        Warp (x,y,w,l) into uvw domain.
        Args:
          boxes_xywlh_yaw: [..., 7] (Tensor or ndarray)
        Returns:
          warped: [..., 7] (Same type as input)
        """
        boxes, is_numpy, orig_shape, orig_dtype = self._prepare_boxes(boxes_xyzwlh_yaw)
        boxes = self._warp_impl(boxes)
        return self._restore_boxes(boxes, is_numpy, orig_shape, orig_dtype)

    def inv_warp_boxes(self, boxes_xyzwlh_yaw):
        """
        Inverse-warp from uvw to xyz.
        Args:
          boxes_xywlh_yaw: [..., 7] (Tensor or ndarray)
        Returns:
          warped: [..., 7] (Same type as input)
        """
        boxes, is_numpy, orig_shape, orig_dtype = self._prepare_boxes(boxes_xyzwlh_yaw)
        boxes = self._inv_warp_impl(boxes)
        return self._restore_boxes(boxes, is_numpy, orig_shape, orig_dtype)
