import torch


class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config, scaler=None):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

        self.scaler = scaler # None => xyz; not None => uvw

    def generate_anchors(self, grid_sizes):
        """
        Args:
            grid_sizes: list of [X, Y] for each head (Z is implied by number of anchor_bottom_heights)

        Returns:
            all_anchors: list of tensors, each with shape [Z, Y, X, A, R, 7], where A is number of anchor sizes,
                         R is number of anchor rotations, 7 represents (x, y, z, dx, dy, dz, yaw)
            num_anchors_per_location: list[int]
        """
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []

        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center \
            in zip(grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            # the center of all grids
            centers = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)

            anchor_size = x_shifts.new_tensor(anchor_size) # [A, 3], A represents the number of anchors of different sizes.
            num_anchor_size = anchor_size.__len__() # Determine how many different sizes of 3D anchor to generate at each grid cell

            if self.scaler is not None:
                uvw_flat = centers.reshape(-1, 3) # [Xu, Yv, Z] -> [N, 3], N = Xu * Yv * Z
                with torch.no_grad():
                    xyz_flat = self.scaler.inv_warp_points(uvw_flat) # [N, 3]
                    r = torch.sqrt(xyz_flat[:, 0]**2 + xyz_flat[:, 1]**2) # [N]
                    s = self.scaler.s_of_r(r).clamp(min=1.0) # [N]
                
                s3 = torch.stack([s, s, torch.ones_like(s)], dim=-1) # 每个网格点anchor尺寸的的缩放系数 [N, 3]
                size_uv = (anchor_size[None, :, :].expand(uvw_flat.size(0), -1, -1) * s3[:, None, :]) # [N, A, 3]
                    
                size_uv = size_uv.view(*centers.shape[:3], num_anchor_size, 3) # [X, Y, Z, A, 3], 每个 grid 上有 A 种不同大小的 anchor
            else:
                size_uv = anchor_size.view(1, 1, 1, num_anchor_size, 3).expand(
                    *centers.shape[:3], num_anchor_size, 3
                )

            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            num_anchor_rotation = anchor_rotation.__len__()

            anchors = centers[:, :, :, None, :].repeat(1, 1, 1, num_anchor_size, 1) # [X, Y, Z, A, 3]
            anchors = torch.cat((anchors, size_uv), dim=-1) # [X, Y, Z, A, 6]

            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)  # [Xu, Yv, Z, A, R, 6]
            rot = anchor_rotation.view(1, 1, 1, 1, -1, 1).expand(*anchors.shape[:5], 1)  # [Xu, Yv, Z, A, R, 1]
            anchors = torch.cat((anchors, rot), dim=-1) # [Xu, Yv, Z, A, R, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous() # [Z, Y, X, A, R, 7]
            anchors[..., 2] += anchors[..., 5] / 2  # 由于 Z 原本是盒子底部的高度，所以要加上高度的一半，变成盒子中心的 Z 坐标
            all_anchors.append(anchors)
        
        return all_anchors, num_anchors_per_location


if __name__ == '__main__':
    from easydict import EasyDict
    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )
    import pdb
    pdb.set_trace()
    A.generate_anchors([[188, 188]])
