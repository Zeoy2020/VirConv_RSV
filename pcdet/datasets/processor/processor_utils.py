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
    Configuration for Radial Scaling Voxelization (RSV).

    r_far : distance (in meters) where the scale nearly decays to 1.
    s_max : maximum scale at r=0 (>=1), e.g., 4.0
    beta  : exponential decay rate (≈1.0 ~ 6.0 typical)
    schedule: exp or linear
    eps   : numerical epsilon
    """
    r_far: float = 50.0
    s_max: float = 2.0
    beta: float = 1.5
    schedule: str = 'exp'
    eps: float = 1e-6

class RadialScalingVoxelization:
    """
    RSV with s(r) decreasing from r=0.

    Exponential schedule:
        s(r) = 1 + (s_max - 1) * exp(-beta * r / r_far)
        -> s(0) = s_max
           s(r_far) ≈ 1 + (s_max - 1)*exp(-beta)
           r > r_far -> saturates to ~1

    Linear schedule:
        s(r) = s_max - (s_max - 1) * r / r_far, when r < r_far
             = 1, otherwise
    """

    def __init__(self, cfg: RadialScaleConfig):
        assert cfg.r_far > 0.0, "r_far must be positive."
        assert cfg.s_max >= 1.0, "s_max must be >= 1."
        self.schedule = self._normalize_schedule(cfg.schedule)
        if self.schedule == 'linear':
            assert cfg.s_max <= 2.0, "linear RSV requires s_max <= 2.0 for a monotonic inverse warp."
        self.cfg = cfg
        self.eps = cfg.eps

    @staticmethod
    def _normalize_schedule(schedule):
        schedule = str(schedule).lower()
        if schedule in ('exp', 'exponential'):
            return 'exp'
        if schedule in ('linear', 'lin'):
            return 'linear'
        raise ValueError(f"Unsupported RSV schedule: {schedule}")

    @staticmethod
    def _euclid_r(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Euclidean distance in XY (no eps inside sqrt)."""
        return np.sqrt(x * x + y * y).astype(np.float32)

    def s_of_r(self, r):
        """
        Radial scale function for the configured RSV schedule.
        """
        r_far = max(self.cfg.r_far, 1e-6)
        beta = float(self.cfg.beta)
        s_max = float(self.cfg.s_max)
        scale_delta = s_max - 1.0

        if isinstance(r, np.ndarray):
            if self.schedule == 'linear':
                s = np.where(r < r_far, s_max - scale_delta * (r / r_far), 1.0)
            else:
                s = 1.0 + scale_delta * np.exp(-beta * (r / r_far))
            return s.astype(np.float32)
        elif torch.is_tensor(r):
            if self.schedule == 'linear':
                linear_s = s_max - scale_delta * (r / r_far)
                s = torch.where(r < r_far, linear_s, torch.ones_like(r))
            else:
                s = 1.0 + scale_delta * torch.exp(-beta * (r / r_far))
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
            if self.schedule == 'linear':
                r = self._linear_inverse_radius_torch(rho)
                x = torch.zeros_like(u, dtype=torch.float32)
                y = torch.zeros_like(v, dtype=torch.float32)
                nonzero = rho > 0
                x[nonzero] = (r[nonzero] / rho[nonzero] * u[nonzero]).float()
                y[nonzero] = (r[nonzero] / rho[nonzero] * v[nonzero]).float()
                x[zero_mask] = 0.0
                y[zero_mask] = 0.0
                return torch.stack([x, y, w], dim=-1)

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
            if self.schedule == 'linear':
                r = self._linear_inverse_radius_numpy(rho)
                x = np.zeros_like(u, dtype=np.float32)
                y = np.zeros_like(v, dtype=np.float32)
                nonzero = rho > 0
                x[nonzero] = (r[nonzero] / rho[nonzero] * u[nonzero]).astype(np.float32)
                y[nonzero] = (r[nonzero] / rho[nonzero] * v[nonzero]).astype(np.float32)
                x[zero_mask] = 0.0
                y[zero_mask] = 0.0
                return np.stack([x, y, w], axis=-1).astype(np.float32)

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

    def _linear_inverse_radius_numpy(self, rho):
        s_max = float(self.cfg.s_max)
        scale_delta = s_max - 1.0
        if scale_delta <= self.eps:
            return rho

        r_far = max(float(self.cfg.r_far), 1e-6)
        r = rho.copy()
        linear_mask = rho <= r_far
        if np.any(linear_mask):
            rho_linear = rho[linear_mask]
            discr = s_max * s_max - 4.0 * scale_delta * rho_linear / r_far
            discr = np.maximum(discr, 0.0)
            r_linear = r_far * (s_max - np.sqrt(discr)) / (2.0 * scale_delta)
            r[linear_mask] = np.clip(r_linear, 0.0, r_far)
        return r

    def _linear_inverse_radius_torch(self, rho):
        s_max = float(self.cfg.s_max)
        scale_delta = s_max - 1.0
        if scale_delta <= self.eps:
            return rho

        r_far = max(float(self.cfg.r_far), 1e-6)
        r = rho.clone()
        linear_mask = rho <= r_far
        if torch.any(linear_mask):
            rho_linear = rho[linear_mask]
            discr = s_max * s_max - 4.0 * scale_delta * rho_linear / r_far
            discr = torch.clamp(discr, min=0.0)
            r_linear = r_far * (s_max - torch.sqrt(discr)) / (2.0 * scale_delta)
            r[linear_mask] = torch.clamp(r_linear, min=0.0, max=r_far)
        return r

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
    def uvw_bounds_from_xy_range(
            self,
            pc_range: Tuple[float, float, float, float, float, float],
            num_boundary_samples: int = 4097,
    ) -> np.ndarray:
        """
        Return a UVW AABB that covers the warped XY rectangle.

        The extrema of u=x*s(r) and v=y*s(r) are not guaranteed to occur at
        rectangle corners. For RSV-linear, the largest |u|/|v| can appear on
        the axis-aligned boundary, so corner-only bounds may clip valid points.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        num_boundary_samples = max(int(num_boundary_samples), 2)
        xs = np.linspace(x_min, x_max, num_boundary_samples, dtype=np.float32)
        ys = np.linspace(y_min, y_max, num_boundary_samples, dtype=np.float32)
        z_ref = np.float32(0.5 * (z_min + z_max))

        boundary_xyz = np.concatenate([
            np.stack([xs, np.full_like(xs, y_min), np.full_like(xs, z_ref)], axis=1),
            np.stack([xs, np.full_like(xs, y_max), np.full_like(xs, z_ref)], axis=1),
            np.stack([np.full_like(ys, x_min), ys, np.full_like(ys, z_ref)], axis=1),
            np.stack([np.full_like(ys, x_max), ys, np.full_like(ys, z_ref)], axis=1),
        ], axis=0)

        uvw = self.warp_points(boundary_xyz)
        u_min, v_min = uvw[:, 0:2].min(axis=0)
        u_max, v_max = uvw[:, 0:2].max(axis=0)
        margin = max(float(self.eps), 1e-4)
        return np.array([
            u_min - margin, v_min - margin, z_min,
            u_max + margin, v_max + margin, z_max
        ], dtype=np.float32)
