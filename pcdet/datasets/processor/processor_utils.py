# ------------------------------------------------------------
#       Radial Scaling Voxelization Utils
# ------------------------------------------------------------
# Radial Scaling Voxelization (RSV) with s(r) decreasing from r=0:
#   r = sqrt(x^2 + y^2)
#   s(0) = s_max  (near -> strongest scaling)
#   s(r_far) ≈ 1  (far  -> no scaling)
#
# Forward (expand near in uv):
#   (u, v) = ( s(r) * x, s(r) * y ),  w = z
# Using a fixed voxel size in (u,v), the world voxel becomes:
#   Δx_world = Δu / s(r), Δy_world = Δv / s(r)
# => near (s large): finer world voxels; far (s≈1): coarser.
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass(frozen=True)
class RadialScaleConfig:
    """
    Configuration for Radial Scaling Voxelization (RSV) with exponential decay.

    r_far : distance (in meters) where the scale nearly decays to 1.
    s_max : maximum scale at r=0 (>=1), e.g., 4.0
    beta  : exponential decay rate (≈1.0 ~ 6.0 typical)
    eps   : numerical epsilon
    """
    r_far: float = 50.0
    s_max: float = 3.0
    beta: float = 1.5
    eps: float = 1e-6

class RadialScalingVoxelization:
    """
    RSV with s(r) exponentially decreasing from r=0 to r=r_far.

    Schedule:
        s(r) = 1 + (s_max - 1) * exp(-beta * r / r_far)
        -> s(0) = s_max
           s(r_far) ≈ 1 + (s_max - 1)*exp(-beta)
           r > r_far -> saturates to ~1
    """

    def __init__(self, cfg: RadialScaleConfig):
        assert cfg.r_far > 0.0, "r_far must be positive."
        assert cfg.s_max >= 1.0, "s_max must be >= 1."
        self.cfg = cfg
        self.eps = cfg.eps

    @staticmethod
    def _euclid_r(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Euclidean distance in XY (no eps inside sqrt)."""
        return np.sqrt(x * x + y * y).astype(np.float32)

    def s_of_r(self, r):
        """
        Exponential decay scale function:
            s(r) = 1 + (s_max - 1) * exp(-beta * r / r_far)
        """
        r_far = max(self.cfg.r_far, 1e-6)
        beta = float(self.cfg.beta)
        s_max = float(self.cfg.s_max)

        if isinstance(r, np.ndarray):
            s = 1.0 + (s_max - 1.0) * np.exp(-beta * (r / r_far))
            return s.astype(np.float32)
        elif torch.is_tensor(r):
            s = 1.0 + (s_max - 1.0) * torch.exp(-beta * (r / r_far))
            return s.to(dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type for r: {type(r)}")

   # -------- Forward (world -> uvw) --------
    def warp_points(self, xyz):
        """
        Forward warp: (x, y, z) -> (u, v, w)
        Supports np.ndarray or torch.Tensor
        """
        is_torch = torch.is_tensor(xyz)
        
        if is_torch:
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            r = torch.sqrt(x * x + y * y)
            s = self.s_of_r(r)
            u, v, w = x * s, y * s, z
            return torch.stack([u, v, w], dim=-1).to(dtype=torch.float32)
        else:
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            r = self._euclid_r(x, y)
            s = self.s_of_r(r)
            u, v, w = x * s, y * s, z
            return np.stack([u, v, w], axis=-1).astype(np.float32)

    # -------- Inverse (uvw -> world) --------
    def inv_warp_points(self, uvw, max_iters=20, tol=1e-6):
        """
        Inverse warp: (u, v, w) -> (x, y, w)
        Supports np.ndarray or torch.Tensor
        """
        is_torch = torch.is_tensor(uvw)

        if is_torch:
            u = uvw[:, 0].double()
            v = uvw[:, 1].double()
            w = uvw[:, 2].float()
            rho = torch.hypot(u, v)
            zero_mask = rho == 0.0
            s_max = self.cfg.s_max
            r_far = self.cfg.r_far
            beta = self.cfg.beta
            r = rho / max(s_max, 1.0)

            for _ in range(max_iters):
                s = 1.0 + (s_max - 1.0) * torch.exp(-beta * r / r_far)
                s_prime = - (s_max - 1.0) * beta / r_far * torch.exp(-beta * r / r_far)
                f = s * r - rho
                fp = s + r * s_prime
                r_new = r - f / (fp + self.eps)
                r_new = torch.clamp(r_new, min=0.0)
                if torch.all(torch.abs(r_new - r) < tol):
                    r = r_new
                    break
                r = r_new

            x = torch.zeros_like(u, dtype=torch.float32)
            y = torch.zeros_like(v, dtype=torch.float32)
            nonzero = rho > 0
            x[nonzero] = (r[nonzero] / rho[nonzero] * u[nonzero]).float()
            y[nonzero] = (r[nonzero] / rho[nonzero] * v[nonzero]).float()
            x[zero_mask] = 0.0
            y[zero_mask] = 0.0

            return torch.stack([x, y, w], dim=-1)
        else:
            u = uvw[:, 0].astype(np.float64)
            v = uvw[:, 1].astype(np.float64)
            w = uvw[:, 2].astype(np.float32)
            rho = np.hypot(u, v)
            zero_mask = (rho == 0.0)
            s_max = self.cfg.s_max
            r_far = self.cfg.r_far
            beta = self.cfg.beta
            r = rho / max(s_max, 1.0)

            for _ in range(max_iters):
                s = 1.0 + (s_max - 1.0) * np.exp(-beta * r / r_far)
                s_prime = - (s_max - 1.0) * beta / r_far * np.exp(-beta * r / r_far)
                f = s * r - rho
                fp = s + r * s_prime
                r_new = r - f / (fp + self.eps)
                r_new = np.maximum(r_new, 0.0)
                if np.all(np.abs(r_new - r) < tol):
                    r = r_new
                    break
                r = r_new

            x = np.zeros_like(u, dtype=np.float32)
            y = np.zeros_like(v, dtype=np.float32)
            nonzero = rho > 0
            x[nonzero] = (r[nonzero] / rho[nonzero] * u[nonzero]).astype(np.float32)
            y[nonzero] = (r[nonzero] / rho[nonzero] * v[nonzero]).astype(np.float32)
            x[zero_mask] = 0.0
            y[zero_mask] = 0.0

            return np.stack([x, y, w], axis=-1).astype(np.float32)

    # -------- Boxes: forward (approx) --------
    def warp_bev_boxes(self, boxes_xywlh_yaw: np.ndarray) -> np.ndarray:
        """
        Approximate forward warp for [x,y,z,w,l,h,yaw]:
          - center: multiply by s(r_center) in XY
          - (w,l):  multiply by s(r_center) (same XY scaling)
          - z,h,yaw: unchanged
        """
        b = boxes_xywlh_yaw.astype(np.float32).copy()
        centers = b[:, :3]
        r = self._euclid_r(centers[:, 0], centers[:, 1], self.cfg.eps)
        s = self.s_of_r(r)

        # center
        b[:, 0] = centers[:, 0] * s
        b[:, 1] = centers[:, 1] * s
        # size in XY
        b[:, 3] = b[:, 3] * s  # width (X)
        b[:, 4] = b[:, 4] * s  # length (Y)
        # z, h, yaw kept
        return b

    # -------- Boxes: inverse (approx) --------
    def inv_warp_bev_boxes(self, boxes_uvwlh_yaw: np.ndarray, iters: int = 6) -> np.ndarray:
        """
        Approximate inverse warp for [u,v,w,w_box,l_box,h,yaw]:
          - center: invert with fixed-point
          - (w,l):  divide by s(r_center_world)
        """
        b = boxes_uvwlh_yaw.astype(np.float32).copy()
        centers_uvw = b[:, :3]
        centers_xyz = self.inv_warp_points(centers_uvw, iters=iters)
        b[:, :3] = centers_xyz

        r = self._euclid_r(centers_xyz[:, 0], centers_xyz[:, 1], self.cfg.eps)
        s = self.s_of_r(r) + self.cfg.eps
        b[:, 3] = b[:, 3] / s
        b[:, 4] = b[:, 4] / s
        return b

    # -------- uvw AABB from world AABB --------
    def uvw_bounds_from_xy_range(self, pc_range: Tuple[float, float, float, float, float, float]) -> np.ndarray:
        """
        Warp 8 corners of world AABB to uvw and return uvw AABB.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        corners = np.array([
            [x_min, y_min, z_min], [x_min, y_min, z_max],
            [x_min, y_max, z_min], [x_min, y_max, z_max],
            [x_max, y_min, z_min], [x_max, y_min, z_max],
            [x_max, y_max, z_min], [x_max, y_max, z_max],
        ], dtype=np.float32)
        uvw = self.warp_points(corners)
        u_min, v_min, w_min = uvw.min(axis=0)
        u_max, v_max, w_max = uvw.max(axis=0)
        return np.array([u_min, v_min, w_min, u_max, v_max, w_max], dtype=np.float32)