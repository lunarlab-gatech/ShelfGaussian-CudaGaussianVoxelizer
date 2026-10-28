import torch
import time
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer
import gc
import numpy as np
from taichi_voxelizer import TaichiVoxelizer

# === splat_into_3d remains the same ===
def splat_into_3d(grid_coords, means3d, opacities, covariances,
                  vol_range, voxel_size, features=None, eps=1e-6):
    num_gaussians = means3d.shape[0]
    grid_density = torch.zeros((*grid_coords.shape[:-1], 1), device=grid_coords.device)
    if features is not None:
        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)), device=grid_coords.device)
    for g in range(num_gaussians):
        sigma = torch.sqrt(torch.diag(covariances[g]))
        factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma)
        bounds = means3d[g, None] + factor * sigma[None]
        if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                ((bounds < vol_range[None, 3:]).max(0).values.min())):
            continue
        bounds = bounds.clamp(vol_range[:3], vol_range[3:])
        bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist()
        slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)])
        # all_lb[g] = torch.tensor([s.start for s in slices], device=grid_coords.device, dtype=torch.int32)
        # all_ub[g] = torch.tensor([s.stop - 1 for s in slices], device=grid_coords.device, dtype=torch.int32)
        diff = grid_coords[slices] - means3d[g]
        maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse() @ diff.unsqueeze(-1)).squeeze(-1)
        density = opacities[g] * torch.exp(-0.5 * maha_dist)
        grid_density[slices] += density
        if features is not None:
            grid_feats[slices] += density * features[g]
    if features is None:
        return grid_density
    grid_feats /= grid_density.clamp(eps)
    return grid_density, grid_feats


# === utility ===
def generate_random_inputs(N):
    means3D = torch.empty((N, 3), device="cuda")
    means3D[:, 0] = torch.rand(N, device="cuda") * 80.0 - 40.0
    means3D[:, 1] = torch.rand(N, device="cuda") * 80.0 - 40.0
    means3D[:, 2] = torch.rand(N, device="cuda") * 6.4 - 1.0
    opacity = torch.rand(N, device="cuda")
    eigenvalues = torch.rand((N, 3), device="cuda") * 7.5 + 0.5
    D = torch.diag_embed(eigenvalues)
    Q, _ = torch.linalg.qr(torch.randn((N, 3, 3), device="cuda"))
    Q[torch.linalg.det(Q) < 0] *= -1.0
    Sigma = Q @ D @ Q.transpose(1, 2) + 1e-4 * torch.eye(3, device="cuda").unsqueeze(0)
    covariances = Sigma.view(N, 9)
    return means3D, opacity, covariances


def build_grid_coords(voxel_range, voxel_size):
    grid_min, grid_max = voxel_range[:3], voxel_range[3:]
    grid_dim = ((grid_max - grid_min) / voxel_size).ceil().long()
    x = torch.arange(grid_dim[0], device="cuda") * voxel_size[0] + grid_min[0]
    y = torch.arange(grid_dim[1], device="cuda") * voxel_size[1] + grid_min[1]
    z = torch.arange(grid_dim[2], device="cuda") * voxel_size[2] + grid_min[2]
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([grid_x, grid_y, grid_z], dim=-1)


# def benchmark_cuda(voxelizer, means, opacity, cov, repeat=5):
#     forward_times = []
#     backward_times = []

#     for _ in range(repeat):
#         m = means.clone().detach().requires_grad_(True)
#         o = opacity.clone().detach().requires_grad_(True)
#         c = cov.clone().detach().requires_grad_(True)

#         torch.cuda.synchronize()
#         t0 = time.time()
#         out = voxelizer(m, o, c)
#         torch.cuda.synchronize()
#         t1 = time.time()
#         loss = out.sum()
#         loss.backward()
#         torch.cuda.synchronize()
#         t2 = time.time()

#         forward_times.append((t1 - t0) * 1000)
#         backward_times.append((t2 - t1) * 1000)

#     return forward_times, backward_times

def taichi_splat_into_3d(
    means3d,          # (G,3)
    opacities,        # (G,1) or (G,)
    covariances,      # (G,3,3)
    vol_range,        # (6,)
    voxel_size,       # float
    features=None,    # (G,C) or None
    eps=1e-6,
):
    if torch.is_tensor(vol_range):
        vol_range_py = vol_range.detach().cpu().flatten().tolist()
    else:
        vol_range_py = list(vol_range)
    voxel_size_f = float(voxel_size)

    means3d     = means3d.contiguous().to(torch.float32)
    covariances = covariances.contiguous().to(torch.float32)

    opacities = opacities.to(torch.float32).contiguous()
    if opacities.ndim == 1:
        opacities = opacities.unsqueeze(1)  # (G,) -> (G,1)

    want_feats = features is not None
    if features is None:
        features = torch.zeros((means3d.size(0), 2),
                               device=means3d.device, dtype=torch.float32)
    else:
        features = features.contiguous().to(torch.float32)
        if features.shape[1] == 1:
            features = torch.cat([features, torch.zeros_like(features)], dim=1)

    means3d_b     = means3d.unsqueeze(0)      # (1,G,3)
    opacities_b   = opacities.unsqueeze(0)    # (1,G,1)
    covariances_b = covariances.unsqueeze(0)  # (1,G,3,3)
    features_b    = features.unsqueeze(0)     # (1,G,>=2)

    vx = TaichiVoxelizer(
        vol_range=vol_range_py,
        voxel_size=voxel_size_f,
        filter_gaussians=False,
        opacity_thresh=0.0,
        eps=eps,
    )

    density_b, feats_b = vx(
        means3d=means3d_b,
        opacities=opacities_b,
        features=features_b,
        covariances=covariances_b,
    )
    # density_b: (1, Nx, Ny, Nz)
    # feats_b:   (1, Nx, Ny, Nz, >=2)

    density = density_b.squeeze(0)       # (Nx,Ny,Nz)
    grid_density = density.unsqueeze(-1) # (Nx,Ny,Nz,1)

    if not want_feats:
        return grid_density

    feats = feats_b.squeeze(0)           # (Nx,Ny,Nz,>=2)
    return grid_density, feats




def benchmark_sw(grid_coords, means, opacity, cov, voxel_range, voxel_size, repeat=3):
    forward_times = []
    backward_times = []

    for _ in range(repeat):
        m = means.clone().detach().requires_grad_(True)
        o = opacity.clone().detach().requires_grad_(True)
        c = cov.view(-1, 3, 3).clone().detach().requires_grad_(True)

        torch.cuda.synchronize()
        t0 = time.time()
        out, *_ = splat_into_3d(grid_coords, m, o, c, voxel_range, voxel_size)
        torch.cuda.synchronize()
        t1 = time.time()
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.time()

        forward_times.append((t1 - t0) * 1000)
        backward_times.append((t2 - t1) * 1000)

    return forward_times, backward_times


# def print_report(name, fwd_times, bwd_times):
#     print(f"\n🧪 {name} benchmark results:")
#     for i, (f, b) in enumerate(zip(fwd_times, bwd_times)):
#         print(f"  Iter {i + 1}: forward={f:.3f} ms, backward={b:.3f} ms, total={f + b:.3f} ms")
#     print(f"  ➤ Avg forward:  {sum(fwd_times) / len(fwd_times):.3f} ms")
#     print(f"  ➤ Avg backward: {sum(bwd_times) / len(bwd_times):.3f} ms")
#     print(f"  ➤ Avg total:    {(sum(fwd_times) + sum(bwd_times)) / len(fwd_times):.3f} ms")

def print_report(name, fwd_times, bwd_times):
    print(f"\n🧪 {name} benchmark results (time in seconds):")
    for i, (f, b) in enumerate(zip(fwd_times, bwd_times)):
        print(f"  Iter {i + 1}: forward = {f:.9f} s, backward = {b:.9f} s, total = {(f + b):.9f} s")
    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)
    avg_total = avg_fwd + avg_bwd
    print(f"  ➤ Avg forward:  {avg_fwd:.9f} s")
    print(f"  ➤ Avg backward: {avg_bwd:.9f} s")
    print(f"  ➤ Avg total:    {avg_total:.9f} s")


def cleanup():
    torch.cuda.empty_cache()  # release the unused cached blocks
    gc.collect()              # clean up on the Python side
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass

if __name__ == "__main__":
    num_gaussians = 18000
    voxel_size_list = [0.4, 0.4, 0.4]
    voxel_range_list = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    voxel_size = torch.tensor(voxel_size_list, device="cuda")
    voxel_range = torch.tensor(voxel_range_list, device="cuda")
    grid_coords = build_grid_coords(voxel_range, voxel_size)

    voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size_list, voxel_range=voxel_range_list)

    repeat = 100
    fwd_cuda, bwd_cuda = [], []
    fwd_sw, bwd_sw = [], []
    fwd_taichi = []

    print(f"🔧 Running benchmark with {num_gaussians} Gaussians, {repeat} unique samples...")

    for i in range(repeat):
        means3D, opacity, covariances = generate_random_inputs(num_gaussians)

        # --- CUDA ---
        m = means3D.clone().detach().requires_grad_(True)
        o = opacity.clone().detach().requires_grad_(True)
        c = covariances.clone().detach().requires_grad_(True)
        torch.cuda.synchronize()
        # t0 = time.time()
        out, _, stats = voxelizer(m, o, c)
        torch.cuda.synchronize()
        # t1 = time.time()
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        # t2 = time.time()
        # fwd_cuda.append((t1 - t0))
        # bwd_cuda.append((t2 - t1))
        fwd_cuda.append(stats[0].item())
        bwd_cuda.append(stats[1].item())

        # cleanup()

        # --- Software ---
        m = means3D.clone().detach().requires_grad_(True)
        o = opacity.clone().detach().requires_grad_(True)
        c = covariances.view(-1, 3, 3).clone().detach().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.time()
        out, *_ = splat_into_3d(grid_coords, m, o, c, voxel_range, voxel_size)
        torch.cuda.synchronize()
        t1 = time.time()
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.time()
        fwd_sw.append((t1 - t0))
        bwd_sw.append((t2 - t1))

        # --- TAICHI FORWARD ONLY ---
        torch.cuda.synchronize()
        taichi_start_time = time.time()
        out, *_ = taichi_splat_into_3d(m, o, c, voxel_range, 0.4)
        torch.cuda.synchronize()
        taichi_end_time = time.time()
        fwd_taichi.append((taichi_end_time - taichi_start_time))
        cleanup()


    print_report("CUDA", fwd_cuda, bwd_cuda)
    print_report("Software", fwd_sw, bwd_sw)

    np.save("fwd_cuda.npy", np.array(fwd_cuda))
    np.save("bwd_cuda.npy", np.array(bwd_cuda))
    np.save("fwd_sw.npy", np.array(fwd_sw))
    np.save("bwd_sw.npy", np.array(bwd_sw))
    
    np.save("fwd_taichi.npy", np.array(fwd_taichi))

    print("✅ Benchmark results saved to npy files.")