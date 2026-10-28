import torch
import torch.nn as nn
# from mmdet3d.registry import MODELS

# from .utils import (apply_to_items, generate_grid, get_covariance,
#                     quat_to_rotmat, unbatched_forward)

def apply_to_items(func, iterable):
    if isinstance(iterable, list):
        return [func(i) for i in iterable]
    elif isinstance(iterable, dict):
        return {k: func(v) for k, v in iterable.items()}

def quat_to_rotmat(quats):
    q = quats / torch.sqrt((quats**2).sum(dim=-1, keepdim=True))
    r, x, y, z = [i.squeeze(-1) for i in q.split(1, dim=-1)]

    R = torch.zeros((*r.shape, 3, 3)).to(r)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def get_covariance(s, r):
    L = torch.zeros((*s.shape[:2], 3, 3)).to(s)
    for i in range(s.size(-1)):
        L[..., i, i] = s[..., i]

    L = r @ L
    covariance = L @ L.mT
    return covariance

def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [*grid_shape, len(grid_shape)]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= val
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(*shape_).expand(*grid_shape)
        grid.append(g)
    return torch.stack(grid, dim=-1)

# def unbatched_forward(func):

#     def wrapper(*args, **kwargs):
#         bs = None
#         for arg in list(args) + list(kwargs.values()):
#             if isinstance(arg, torch.Tensor):
#                 if bs is None:
#                     bs = arg.size(0)
#                 else:
#                     assert bs == arg.size(0)

#         outputs = []
#         for i in range(bs):
#             output = func(
#                 *[
#                     arg[i] if isinstance(arg, torch.Tensor) else arg
#                     for arg in args
#                 ], **{
#                     k: v[i] if isinstance(v, torch.Tensor) else v
#                     for k, v in kwargs.items()
#                 })
#             outputs.append(output)

#         if isinstance(outputs[0], tuple):
#             return tuple([
#                 torch.stack([out[i] for out in outputs])
#                 for i in range(len(outputs[0]))
#             ])
#         else:
#             return torch.stack(outputs)

#     return wrapper


def splat_into_3d(grid_coords,
                  means3d,
                  opacities,
                  covariances,
                  vol_range,
                  voxel_size,
                  features=None,
                  eps=1e-6):
    vol_range = vol_range.to(means3d.device)
    grid_coords = grid_coords.to(means3d.device)
    grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                               device=grid_coords.device)
    if features is not None:
        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                 device=grid_coords.device)

    for g in range(means3d.size(0)):
        sigma = torch.sqrt(torch.diag(covariances[g]))
        factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma)
        bounds = means3d[g, None] + factor * sigma[None]
        if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                ((bounds < vol_range[None, 3:]).max(0).values.min())):
            continue
        bounds = bounds.clamp(vol_range[:3], vol_range[3:])
        bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist()
        slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)])

        diff = grid_coords[slices] - means3d[g]
        maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse()
                     @ diff.unsqueeze(-1)).squeeze(-1)
        density = opacities[g] * torch.exp(-0.5 * maha_dist)
        grid_density[slices] += density
        if features is not None:
            grid_feats[slices] += density * features[g]

    if features is None:
        return grid_density
    grid_feats /= grid_density.clamp(eps)
    return grid_density, grid_feats


# @MODELS.register_module()
class GaussianVoxelizer(nn.Module):
# class GaussianVoxelizer():

    def __init__(self,
                 vol_range,
                 voxel_size,
                 filter_gaussians=False,
                 opacity_thresh=0,
                 covariance_thresh=0):
        super().__init__()
        self.voxel_size = voxel_size
        vol_range = torch.tensor(vol_range)
        self.register_buffer('vol_range', vol_range)

        self.grid_shape = ((vol_range[3:] - vol_range[:3]) /
                           voxel_size).int().tolist()
        grid_coords = generate_grid(self.grid_shape, offset=0.5)
        grid_coords = grid_coords * voxel_size + vol_range[:3]
        self.register_buffer('grid_coords', grid_coords)

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.covariance_thresh = covariance_thresh

    # @unbatched_forward
    def forward(self,
                means3d,
                opacities,
                covariances=None,
                scales=None,
                rotations=None,
                **kwargs):
        if covariances is None:
            covariances = get_covariance(scales, quat_to_rotmat(rotations))
        gaussians = dict(
            means3d=means3d,
            opacities=opacities,
            covariances=covariances,
            **kwargs)

        if self.filter_gaussians:
            mask = opacities.squeeze(1) > self.opacity_thresh
            for i in range(3):
                mask &= (means3d[:, i] >= self.vol_range[i]) & (
                    means3d[:, i] <= self.vol_range[i + 3])
            if self.covariance_thresh > 0:
                cov_diag = torch.diagonal(covariances, dim1=1, dim2=2)
                mask &= ((cov_diag.min(1)[0] * 6) > self.covariance_thresh)
            gaussians = apply_to_items(lambda x: x[mask], gaussians)

        return splat_into_3d(
            self.grid_coords,
            **gaussians,
            vol_range=self.vol_range,
            voxel_size=self.voxel_size)