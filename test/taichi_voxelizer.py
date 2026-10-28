# import taichi as ti
# import torch

# ti.init(arch=ti.gpu)

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

# def tensor_to_field(tensor):
#     assert tensor.dim() in (2, 3)
#     if tensor.dim() == 2:
#         n, c = tensor.shape
#         if c == 1:
#             field = ti.field(dtype=ti.f32, shape=n)
#             tensor = tensor.squeeze(1)
#         else:
#             field = ti.Vector.field(c, dtype=ti.f32, shape=n)
#     else:
#         n, c1, c2 = tensor.shape
#         field = ti.Matrix.field(c1, c2, dtype=ti.f32, shape=n)
#     field.from_torch(tensor)
#     return field


# @ti.data_oriented
# class TaichiVoxelizer:

#     def __init__(self,
#                  vol_range,
#                  voxel_size,
#                  filter_gaussians=False,
#                  opacity_thresh=0,
#                  eps=1e-6):
#         self.vol_range = vol_range
#         self.voxel_size = voxel_size
#         self.grid_shape = [
#             int((vol_range[i + 3] - vol_range[i]) / voxel_size)
#             for i in range(3)
#         ]

#         self.filter_gaussians = filter_gaussians
#         self.opacity_thresh = opacity_thresh
#         self.eps = eps
#         self.is_inited = False

#     def init_fields(self, dims):
#         self.density_field = ti.field(dtype=ti.f32, shape=self.grid_shape)
#         self.feature_field = ti.Vector.field(
#             dims, dtype=ti.f32, shape=self.grid_shape)
#         self.weight_accum = ti.field(dtype=ti.f32, shape=self.grid_shape)
#         self.feature_accum = ti.Vector.field(
#             dims, dtype=ti.f32, shape=self.grid_shape)
#         self.is_inited = True

#     def reset_fields(self):
#         self.density_field.fill(0)
#         self.feature_field.fill(0)
#         self.weight_accum.fill(0)
#         self.feature_accum.fill(0)

#     @ti.kernel
#     def voxelize(self, positions: ti.template(), opacities: ti.template(),
#                  features: ti.template(), covariances: ti.template()):
#         for g in range(positions.shape[0]):
#             pos = positions[g]
#             opac = opacities[g]
#             feat = features[g]
#             cov = covariances[g]
#             cov_inv = cov.inverse()

#             sigma = ti.sqrt(ti.Vector([cov[0, 0], cov[1, 1], cov[2, 2]]))
#             min_bound = pos - sigma * 3
#             max_bound = pos + sigma * 3

#             min_indices, max_indices = [0] * 3, [0] * 3
#             for i in ti.static(range(3)):
#                 min_indices[i] = ti.max(
#                     ti.cast(
#                         (min_bound[i] - self.vol_range[i]) / self.voxel_size,
#                         ti.i32), 0)
#                 max_indices[i] = ti.min(
#                     ti.cast(
#                         (max_bound[i] - self.vol_range[i]) / self.voxel_size,
#                         ti.i32), self.grid_shape[i] - 1)

#             for i, j, k in ti.ndrange(*[(min_indices[i], max_indices[i] + 1)
#                                         for i in ti.static(range(3))]):
#                 voxel_center = (
#                     ti.Vector([i, j, k]) * self.voxel_size +
#                     ti.Vector(self.vol_range[:3]) + 0.5)

#                 delta = voxel_center - pos
#                 exponent = -0.5 * delta.dot(cov_inv @ delta)
#                 contrib = ti.exp(exponent) * opac

#                 self.weight_accum[i, j, k] += contrib
#                 self.feature_accum[i, j, k] += feat * contrib

#     @ti.kernel
#     def normalize(self):
#         for i, j, k in self.feature_accum:
#             if self.weight_accum[i, j, k] > self.eps:
#                 self.feature_field[i, j, k] = self.feature_accum[
#                     i, j, k] / self.weight_accum[i, j, k]
#                 self.density_field[i, j, k] = self.weight_accum[i, j, k]
#             else:
#                 self.feature_field[i, j, k] = ti.Vector(
#                     [0.0 for i in range(self.feature_field[i, j, k].n)])
#                 self.density_field[i, j, k] = 0.0

#     @unbatched_forward
#     def __call__(self, **gaussians):
#         if self.filter_gaussians:
#             assert False  # slower, don't know why
#             mask = gaussians['opacities'][:, 0] > self.opacity_thresh
#             for i in range(3):
#                 mask &= (gaussians['means3d'][:, i] >= self.vol_range[i]) & (
#                     gaussians['means3d'][:, i] <= self.vol_range[i + 3])
#             gaussians = apply_to_items(lambda x: x[mask], gaussians)

#         # if 'covariances' not in gaussians:
#         #     gaussians['covariances'] = get_covariance(
#         #         gaussians.pop('scales'),
#         #         quat_to_rotmat(gaussians.pop('rotations')))

#         device = gaussians['means3d'].device
#         gaussians = {k: tensor_to_field(v) for k, v in gaussians.items()}

#         if not self.is_inited:
#             self.init_fields(gaussians['features'].n)
#         else:
#             self.reset_fields()

#         self.voxelize(gaussians['means3d'], gaussians['opacities'],
#                       gaussians['features'], gaussians['covariances'])
#         self.normalize()
#         return (self.density_field.to_torch(device),
#                 self.feature_field.to_torch(device))

import taichi as ti
import torch
# from .utils import (apply_to_items, get_covariance, quat_to_rotmat,
#                     unbatched_forward)

ti.init(arch=ti.gpu)

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

def unbatched_forward(func):

    def wrapper(*args, **kwargs):
        bs = None
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if bs is None:
                    bs = arg.size(0)
                else:
                    assert bs == arg.size(0)

        outputs = []
        for i in range(bs):
            output = func(
                *[
                    arg[i] if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ], **{
                    k: v[i] if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                })
            outputs.append(output)

        if isinstance(outputs[0], tuple):
            return tuple([
                torch.stack([out[i] for out in outputs])
                for i in range(len(outputs[0]))
            ])
        else:
            return torch.stack(outputs)

    return wrapper

def tensor_to_field(tensor):
    assert tensor.dim() in (2, 3)
    if tensor.dim() == 2:
        n, c = tensor.shape
        if c == 1:
            field = ti.field(dtype=ti.f32, shape=n)
            tensor = tensor.squeeze(1)
        else:
            field = ti.Vector.field(c, dtype=ti.f32, shape=n)
    else:
        n, c1, c2 = tensor.shape
        field = ti.Matrix.field(c1, c2, dtype=ti.f32, shape=n)
    field.from_torch(tensor)
    return field


@ti.data_oriented
class TaichiVoxelizer:

    def __init__(self,
                 vol_range,
                 voxel_size,
                 filter_gaussians=False,
                 opacity_thresh=0,
                 eps=1e-6):
        self.vol_range = vol_range
        self.voxel_size = voxel_size
        self.grid_shape = [
            int((vol_range[i + 3] - vol_range[i]) / voxel_size)
            for i in range(3)
        ]

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.eps = eps
        self.is_inited = False

    def init_fields(self, dims):
        self.density_field = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.feature_field = ti.Vector.field(
            dims, dtype=ti.f32, shape=self.grid_shape)
        self.weight_accum = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.feature_accum = ti.Vector.field(
            dims, dtype=ti.f32, shape=self.grid_shape)
        self.is_inited = True

    def reset_fields(self):
        self.density_field.fill(0)
        self.feature_field.fill(0)
        self.weight_accum.fill(0)
        self.feature_accum.fill(0)

    @ti.kernel
    def voxelize(self, positions: ti.template(), opacities: ti.template(),
                 features: ti.template(), covariances: ti.template()):
        for g in range(positions.shape[0]):
            pos = positions[g]
            opac = opacities[g]
            feat = features[g]
            cov = covariances[g]
            cov_inv = cov.inverse()

            sigma = ti.sqrt(ti.Vector([cov[0, 0], cov[1, 1], cov[2, 2]]))
            min_bound = pos - sigma * 3
            max_bound = pos + sigma * 3

            min_indices, max_indices = [0] * 3, [0] * 3
            for i in ti.static(range(3)):
                min_indices[i] = ti.max(
                    ti.cast(
                        (min_bound[i] - self.vol_range[i]) / self.voxel_size,
                        ti.i32), 0)
                max_indices[i] = ti.min(
                    ti.cast(
                        (max_bound[i] - self.vol_range[i]) / self.voxel_size,
                        ti.i32), self.grid_shape[i] - 1)

            for i, j, k in ti.ndrange(*[(min_indices[i], max_indices[i] + 1)
                                        for i in ti.static(range(3))]):
                voxel_center = (
                    (ti.Vector([i, j, k])+ 0.5) * self.voxel_size +
                    ti.Vector(self.vol_range[:3]))

                delta = voxel_center - pos
                exponent = -0.5 * delta.dot(cov_inv @ delta)
                contrib = ti.exp(exponent) * opac

                self.weight_accum[i, j, k] += contrib
                self.feature_accum[i, j, k] += feat * contrib

    @ti.kernel
    def normalize(self):
        for i, j, k in self.feature_accum:
            if self.weight_accum[i, j, k] > self.eps:
                self.feature_field[i, j, k] = self.feature_accum[
                    i, j, k] / self.weight_accum[i, j, k]
                self.density_field[i, j, k] = self.weight_accum[i, j, k]
            else:
                self.feature_field[i, j, k] = ti.Vector(
                    [0.0 for i in range(self.feature_field[i, j, k].n)])
                self.density_field[i, j, k] = 0.0

    @unbatched_forward
    def __call__(self, **gaussians):
        if self.filter_gaussians:
            assert False  # slower, don't know why
            mask = gaussians['opacities'][:, 0] > self.opacity_thresh
            for i in range(3):
                mask &= (gaussians['means3d'][:, i] >= self.vol_range[i]) & (
                    gaussians['means3d'][:, i] <= self.vol_range[i + 3])
            gaussians = apply_to_items(lambda x: x[mask], gaussians)

        if 'covariances' not in gaussians:
            gaussians['covariances'] = get_covariance(
                gaussians.pop('scales'),
                quat_to_rotmat(gaussians.pop('rotations')))

        device = gaussians['means3d'].device
        gaussians = {k: tensor_to_field(v) for k, v in gaussians.items()}

        if not self.is_inited:
            self.init_fields(gaussians['features'].n)
        else:
            self.reset_fields()

        self.voxelize(gaussians['means3d'], gaussians['opacities'],
                      gaussians['features'], gaussians['covariances'])
        self.normalize()
        return (self.density_field.to_torch(device),
                self.feature_field.to_torch(device))