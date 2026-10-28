"""
Check the CUDA voxelizer against a brute-force float64 PyTorch reference.

    python test/test_reference.py

No external data or Taichi needed. Two checks:
  1. Forward: 18000 Gaussians, 128-D features (4 channel blocks of 32), density and grid features.
  2. Forward + backward: 500 Gaussians inside the range + 50 centred outside it, 40-D features,
     random upstream gradients, gradients of means3D / opacity / covariances / features.
The reference uses the 3-sigma voxel box of each Gaussian returned by the forward, so the
comparison checks the splatting and gradient math of the tile kernels.
"""
import sys

import torch
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer, _C

DEVICE = "cuda"
VOXEL_SIZE = [0.4, 0.4, 0.4]
VOXEL_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
EPS = 1e-6      # same clamp as the forward: F_v = sum_g w_gv f_g / max(D_v, eps)
RTOL = 1e-4     # relative L2 error allowed (float32 + fast math vs float64)

lo = torch.tensor(VOXEL_RANGE[:3], device=DEVICE, dtype=torch.float64)
hi = torch.tensor(VOXEL_RANGE[3:], device=DEVICE, dtype=torch.float64)
vs = torch.tensor(VOXEL_SIZE, device=DEVICE, dtype=torch.float64)
X, Y, Z = [int(round(v)) for v in ((hi - lo) / vs).tolist()]


def generate_inputs(num_gaussians, feat_dim, num_outside=0, seed=0):
    """Random Gaussians in the voxel range (+ num_outside centred beyond x_max)."""
    torch.manual_seed(seed)
    lo32, hi32 = lo.float(), hi.float()
    means3D = torch.rand(num_gaussians, 3, device=DEVICE) * (hi32 - lo32) + lo32
    if num_outside:
        outside = torch.rand(num_outside, 3, device=DEVICE) * (hi32 - lo32) + lo32
        outside[:, 0] = hi32[0] + torch.rand(num_outside, device=DEVICE) * 1.5
        means3D = torch.cat([means3D, outside])
    n = means3D.shape[0]
    opacity = torch.rand(n, device=DEVICE)
    R, _ = torch.linalg.qr(torch.randn(n, 3, 3, device=DEVICE))
    var = torch.rand(n, 3, device=DEVICE) * 2.0 + 0.5
    covariances = (R @ torch.diag_embed(var) @ R.mT).reshape(n, 9)
    features = torch.randn(n, feat_dim, device=DEVICE)
    return [t.contiguous().requires_grad_() for t in (means3D, opacity, covariances, features)]


def voxel_boxes(means3D, opacity, covariances, features):
    """Inclusive voxel index box [lb, ub] of each Gaussian, as computed by the forward."""
    out = _C.gaussian_voxelizer_forward(
        means3D.detach(), opacity.detach(), covariances.detach(),
        torch.tensor(VOXEL_SIZE, device=DEVICE), torch.tensor(VOXEL_RANGE, device=DEVICE),
        features.detach())
    return out[2].long(), out[3].long()


def reference_voxelize(means3D, opacity, covariances, features, lb, ub, chunk=64):
    """Brute-force density (X,Y,Z) and grid features (X,Y,Z,C) in float64, differentiable."""
    n, C = features.shape
    m, o, f = means3D.double(), opacity.double(), features.double()
    cov_inv = torch.linalg.inv(covariances.double().view(n, 3, 3))
    extent = (ub - lb + 1).clamp_min(0)
    offsets = torch.stack(torch.meshgrid(
        *[torch.arange(k, device=DEVICE) for k in extent.max(0).values.tolist()], indexing="ij"), -1).view(-1, 3)

    density = torch.zeros(X * Y * Z, device=DEVICE, dtype=torch.float64)
    feats_sum = torch.zeros(X * Y * Z, C, device=DEVICE, dtype=torch.float64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        g_local, p = (offsets[None] < extent[s:e, None]).all(-1).nonzero(as_tuple=True)
        g = g_local + s
        idx = lb[g] + offsets[p]                                   # voxel index of every (Gaussian, voxel) pair
        d = lo + (idx + 0.5) * vs - m[g]                           # voxel centre - mean
        w = o[g] * torch.exp(-0.5 * torch.einsum("ki,kij,kj->k", d, cov_inv[g], d))
        lin = (idx[:, 0] * Y + idx[:, 1]) * Z + idx[:, 2]
        density = density.index_add(0, lin, w)
        feats_sum = feats_sum.index_add(0, lin, w[:, None] * f[g])
    grid_feats = feats_sum / density.clamp_min(EPS)[:, None]
    return density.view(X, Y, Z), grid_feats.view(X, Y, Z, C)


def rel_err(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def report(name, err):
    ok = err < RTOL
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<24} rel. err = {err:.2e}")
    return ok


def test_forward(voxelizer, num_gaussians=18000, feat_dim=128):
    print(f"[forward] {num_gaussians} Gaussians, {feat_dim}-D features")
    inputs = generate_inputs(num_gaussians, feat_dim)
    with torch.no_grad():
        density, grid_feats, _ = voxelizer(*inputs)
        lb, ub = voxel_boxes(*inputs)
        density_ref, grid_feats_ref = reference_voxelize(*inputs, lb, ub, chunk=32)
    return all([report("density", rel_err(density, density_ref)),
                report("grid_feats", rel_err(grid_feats, grid_feats_ref))])


def test_backward(voxelizer, num_gaussians=500, num_outside=50, feat_dim=40):
    print(f"[backward] {num_gaussians} Gaussians + {num_outside} outside the range, {feat_dim}-D features")
    inputs = generate_inputs(num_gaussians, feat_dim, num_outside=num_outside, seed=1)
    torch.manual_seed(2)
    grad_density = torch.randn(X, Y, Z, device=DEVICE)
    grad_feats = torch.randn(X, Y, Z, feat_dim, device=DEVICE)

    density, grid_feats, _ = voxelizer(*inputs)
    ((density * grad_density).sum() + (grid_feats * grad_feats).sum()).backward()

    inputs_ref = [t.detach().clone().requires_grad_() for t in inputs]
    lb, ub = voxel_boxes(*inputs)
    density_ref, grid_feats_ref = reference_voxelize(*inputs_ref, lb, ub)
    ((density_ref * grad_density.double()).sum() + (grid_feats_ref * grad_feats.double()).sum()).backward()

    outside = slice(num_gaussians, num_gaussians + num_outside)
    oks = [report("density", rel_err(density, density_ref)),
           report("grid_feats", rel_err(grid_feats, grid_feats_ref))]
    for name, t, t_ref in zip(("means3D", "opacity", "covariances", "features"), inputs, inputs_ref):
        oks.append(report(f"grad {name}", rel_err(t.grad, t_ref.grad)))
    oks.append(report("grad means3D (outside)", rel_err(inputs[0].grad[outside], inputs_ref[0].grad[outside])))
    return all(oks)


if __name__ == "__main__":
    voxelizer = PyCudaGaussianVoxelizer(voxel_size=VOXEL_SIZE, voxel_range=VOXEL_RANGE)
    ok = test_forward(voxelizer)
    torch.cuda.empty_cache()
    ok = test_backward(voxelizer) and ok
    print("All checks passed." if ok else "Some checks FAILED.")
    sys.exit(0 if ok else 1)
