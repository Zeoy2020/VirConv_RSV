# pcdet/models/backbones_3d/modules/rsv_invwarp_torch.py
import torch
import torch.nn as nn
from ....utils.spconv_utils import spconv
import math

import torch
import torch.nn as nn

class RadialScaleTorch(nn.Module):
    """
    Radial Scaling in Torch, consistent with RadialScalingVoxelization (RSV).

    s(r) = 1 + (s_max - 1) * exp(-beta * r / r_far)
    Forward: (x,y,z) -> (u,v,w) by multiplying s(r) in XY
    Inverse: fixed-point iteration to recover (x,y) from (u,v)
    """

    def __init__(self, r_far=50.0, s_max=3.0, beta=1.5, eps=1e-6):
        super().__init__()
        self.r_far = float(r_far)
        self.s_max = float(s_max)
        self.beta = float(beta)
        self.eps = float(eps)
        assert self.r_far > 0 and self.s_max >= 1.0

    def s_of_r(self, r: torch.Tensor) -> torch.Tensor:
        """Exponential decay scale: s(r) = 1 + (s_max - 1) * exp(-beta * r / r_far)"""
        return 1.0 + (self.s_max - 1.0) * torch.exp(-self.beta * r / max(self.r_far, 1e-6))

    @torch.no_grad()
    def warp_points(self, xyz: torch.Tensor) -> torch.Tensor:
        """Forward warp: (x, y, z) -> (u, v, w)"""
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = torch.sqrt(x * x + y * y)
        s = self.s_of_r(r)
        u = x * s
        v = y * s
        w = z
        return torch.stack([u, v, w], dim=-1)

    @torch.no_grad()
    def inv_warp_points(self, uvw: torch.Tensor, max_iters: int = 20, tol: float = 1e-6) -> torch.Tensor:
        """
        Inverse warp: (u,v,w) -> (x,y,w) using fixed-point iteration
        Stable version similar to RSV's implementation.
        """
        u, v, w = uvw[:, 0].double(), uvw[:, 1].double(), uvw[:, 2].float()
        rho = torch.hypot(u, v)
        zero_mask = rho == 0.0

        # initialize r with upper bound
        r = rho / max(self.s_max, 1.0)

        for _ in range(max_iters):
            s = 1.0 + (self.s_max - 1.0) * torch.exp(-self.beta * r / max(self.r_far, 1e-6))
            f = s * r - rho
            # derivative: s + r * s' = s - r * (s_max-1)*beta/r_far*exp(-beta*r/r_far)
            s_prime = - (self.s_max - 1.0) * self.beta / max(self.r_far, 1e-6) * torch.exp(-self.beta * r / max(self.r_far, 1e-6))
            fp = s + r * s_prime + self.eps
            r_new = r - f / fp
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
        return torch.stack([x, y, w], dim=-1)


@torch.no_grad()
def remap_sparse_uvw_to_world(
    sp_uvw: spconv.SparseConvTensor,
    scaler: RadialScaleTorch,
    uvw_range,              # (umin,vmin,wmin,umax,vmax,wmax)
    voxel_size_uv,          # (du, dv, dw) in uvw space, BEFORE strides
    world_range,            # (xmin,ymin,zmin,xmax,ymax,zmax)
    voxel_size_world,       # (dx, dy, dz) fixed world voxel size
    xy_total_stride: int=1,
    z_total_stride: int=1,
    reduce: str='mean'
) -> spconv.SparseConvTensor:
    """
    Convert a SparseConvTensor defined on a uniform uvw grid
    back to a uniform Cartesian (x,y,z) world grid with *kept stride* by:
      1) computing each uvw voxel center in metric units,
      2) inverse-warping centers to xyz via scaler.inv_warp_points,
      3) Re-hash to world voxel indices using *stride-compensated* world voxel size:
         dx_eff = dx * xy_total_stride, dy_eff = dy * xy_total_stride, dz_eff = dz * z_total_stride
      4) re-hashing to world voxel indices and aggregating features.

    Args:
        sp_uvw: SparseConvTensor on uvw grid.
            features: [M, C]
            indices : [M, 4] = [b, z(w), y(v), x(u)]
        scaler: RadialScaleTorch, must provide inv_warp_points(tensor[N,3])->tensor[N,3]
        uvw_range: float6, metric AABB in uvw: (umin,vmin,wmin,umax,vmax,wmax)
        voxel_size_uv: float3, base uvw voxel size BEFORE any stride (du,dv,dw)
        world_range: float6, metric AABB in xyz: (xmin,ymin,zmin,xmax,ymax,zmax)
        voxel_size_world: float3, base world voxel size (dx,dy,dz)
        xy_total_stride: effective stride along u/v (e.g., 8 if backbone downsample x8)
        z_total_stride : effective stride along w   (e.g., 1 if no z downsample)
        reduce: how to aggregate features landing in the same world voxel.

    Returns:
        SparseConvTensor on xyz grid:
            features: [U, C]
            indices : [U, 4] = [b, z, y, x] (world grid)
            spatial_shape = [Z, Y, X]
            batch_size    = sp_uvw.batch_size
    """
    assert isinstance(sp_uvw, spconv.SparseConvTensor)
    device = sp_uvw.features.device
    feats = sp_uvw.features                    # [M, C]
    inds  = sp_uvw.indices.int()               # [M, 4] = [b, z, y, x]

    # Effective uvw voxel size at the current tensor stride
    du, dv, dw = [float(x) for x in voxel_size_uv]
    du_eff, dv_eff, dw_eff = du * xy_total_stride, dv * xy_total_stride, dw * z_total_stride

    # uvw range
    umin, vmin, wmin, umax, vmax, wmax = [float(x) for x in uvw_range]

    # Convert voxel indices -> uvw centers (metric)
    x_idx = inds[:, 3].to(torch.float32)
    y_idx = inds[:, 2].to(torch.float32)
    z_idx = inds[:, 1].to(torch.float32)
    u = umin + (x_idx + 0.5) * du_eff
    v = vmin + (y_idx + 0.5) * dv_eff
    w = wmin + (z_idx + 0.5) * dw_eff
    uvw_center = torch.stack([u, v, w], dim=-1).to(device=device, dtype=feats.dtype)

    # Inverse warp centers to world xyz
    xyz_center = scaler.inv_warp_points(uvw_center)  # [M,3], metric world coords

    # stride-compensated world grid
    dx, dy, dz = [float(x) for x in voxel_size_world]
    # keep stride by enlarging world voxel size:
    dx_eff = dx * xy_total_stride
    dy_eff = dy * xy_total_stride
    dz_eff = dz * z_total_stride

    xmin, ymin, zmin, xmax, ymax, zmax = [float(x) for x in world_range]
    X = round((xmax - xmin) / dx_eff)
    Y = round((ymax - ymin) / dy_eff)
    Z = round((zmax - zmin) / dz_eff)

    ix = torch.floor((xyz_center[:, 0] - xmin) / dx_eff).long()
    iy = torch.floor((xyz_center[:, 1] - ymin) / dy_eff).long()
    iz = torch.floor((xyz_center[:, 2] - zmin) / dz_eff).long()
    b  = inds[:,0]

    valid = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
    if not torch.any(valid):
        return spconv.SparseConvTensor(
            features=feats.new_zeros((0, feats.shape[1])),
            indices=inds.new_zeros((0, 4)),
            spatial_shape=[Z, Y, X],    # keep the world spatial shape even if empty
            batch_size=sp_uvw.batch_size
        )

    ix, iy, iz, b = ix[valid], iy[valid], iz[valid], b[valid]
    feats_valid   = feats[valid]
    world_keys    = torch.stack([b, iz, iy, ix], dim=1)  # [K,4]

    # Aggregate collisions (mean/sum/max)
    uniq, inv = torch.unique(world_keys, dim=0, return_inverse=True)
    C = feats_valid.shape[1]
    if reduce == 'mean':
        sums = torch.zeros((uniq.shape[0], C), device=device, dtype=feats_valid.dtype)
        cnts = torch.zeros((uniq.shape[0], 1), device=device, dtype=feats_valid.dtype)
        sums.index_add_(0, inv, feats_valid)
        cnts.index_add_(0, inv, torch.ones((feats_valid.shape[0], 1), device=device, dtype=feats_valid.dtype))
        feats_out = sums / torch.clamp(cnts, min=1.0)

    elif reduce == 'sum':
        feats_out = torch.zeros((uniq.shape[0], C), device=device, dtype=feats_valid.dtype)
        feats_out.index_add_(0, inv, feats_valid)

    elif reduce == 'max':
        # Use scatter_reduce if available (PyTorch >= 2.0), else fallback to grouped max
        if hasattr(torch, "scatter_reduce"):
            feats_out = torch.full((uniq.shape[0], C), -1e9, device=device, dtype=feats_valid.dtype)
            feats_out = feats_out.scatter_reduce(0, inv.unsqueeze(-1).expand_as(feats_valid),
                                                 feats_valid, reduce="amax", include_self=True)
        else:
            feats_out = torch.full((uniq.shape[0], C), -1e9, device=device, dtype=feats_valid.dtype)
            # simple (slower) fallback
            for k in range(uniq.shape[0]):
                mask = (inv == k)
                if mask.any():
                    feats_out[k] = torch.max(feats_valid[mask], dim=0).values
    else:
        raise ValueError("reduce must be 'mean' | 'sum' | 'max'")

    return spconv.SparseConvTensor(
        features=feats_out,
        indices=uniq.int(),               # [U,4] = [b,z,y,x]
        spatial_shape=[Z, Y, X],
        batch_size=sp_uvw.batch_size
    )