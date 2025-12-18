# pcdet/models/box_adapters/rsv_box_adapter.py
import torch
from pcdet.utils import common_utils


class RSVBoxAdapter:
    """
    Adapter that warps 3D boxes between Cartesian (xyz) and radial-scaled (uvw) spaces.
    - We only scale BEV-related dimensions (x, y, w, l) with the same scale s(r) used for points.
    - z, h, yaw remain unchanged (you can extend if needed).

    Assumptions:
      boxes format: [..., 7] = [x, y, z, w, l, h, yaw]
      scaler: provides s_of_r(r: Tensor)->Tensor and inv_warp_points(xyz: Tensor[N,3])->Tensor[N,3]
    """
    def __init__(self, scaler):
        self.scaler = scaler

    @torch.no_grad()
    def warp_boxes(self, boxes_xyzwlh_yaw: torch.Tensor) -> torch.Tensor:
        """
        Warp (x,y,w,l) into uvw domain by applying the same radial scale s(r) used for points.
        z, h, yaw are kept unchanged.
        Args:
          boxes_xywlh_yaw: [..., 7]
        Returns:
          warped: [..., 7] in uvw-like space
        """
        b = boxes_xyzwlh_yaw.clone().float()
        x, y = b[..., 0], b[..., 1]
        r = torch.sqrt(x * x + y * y + getattr(self.scaler, "eps", 1e-6))
        s = self.scaler.s_of_r(r)  # scale factor per radius
        b[..., 0] = x * s
        b[..., 1] = y * s
        b[..., 3] = b[..., 3] * s  # w
        b[..., 4] = b[..., 4] * s  # l
        return b

    @torch.no_grad()
    def inv_warp_boxes(self, boxes_xyzwlh_yaw: torch.Tensor) -> torch.Tensor:
        """
        Inverse-warp from uvw to xyz for boxes (optional; only needed if you decode in uvw then want xyz).
        - Centers (x,y,z) use inv_warp_points.
        - Width/length divide by s(r) evaluated at the inverse-warped center.
        """
        b = boxes_xyzwlh_yaw.clone().float()
        # inverse-warp center to xyz
        xyz = self.scaler.inv_warp_points(b[..., 0:3].reshape(-1, 3))
        # self.logger.debug(f"Inv-warping {xyz.shape} boxes")
        xyz = xyz.reshape(b[..., 0:3].shape)
        b[..., 0:3] = xyz
        x, y = xyz[..., 0], xyz[..., 1]
        r = torch.sqrt(x * x + y * y + getattr(self.scaler, "eps", 1e-6))
        s = self.scaler.s_of_r(r)
        b[..., 3] = b[..., 3] / s
        b[..., 4] = b[..., 4] / s
        return b