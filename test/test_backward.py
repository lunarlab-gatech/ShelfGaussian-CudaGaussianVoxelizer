import torch
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer
import time
from taichi_voxelizer import TaichiVoxelizer
from gaussian_voxelizer import GaussianVoxelizer
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



def generate_random_covariances_with_rotation(num_gaussians, min_scale=0.5, max_scale=8.0, epsilon=1e-4):
    """
    Generate symmetric positive-definite covariance matrices:
        Σ = Q D Qᵀ + εI
    where D = diag(scale), Q ∈ SO(3)
    Output: (N, 9) flattened 3x3 covariance matrices
    """
    # 1. Sample random eigenvalues (positive scales)
    eigenvalues = torch.rand((num_gaussians, 3)).cuda()
    eigenvalues = eigenvalues * (max_scale - min_scale) + min_scale

    # 2. Construct diagonal matrix D
    D = torch.zeros((num_gaussians, 3, 3)).cuda()
    D[:, 0, 0] = eigenvalues[:, 0]
    D[:, 1, 1] = eigenvalues[:, 1]
    D[:, 2, 2] = eigenvalues[:, 2]

    # 3. Generate random orthogonal matrix Q via QR
    A = torch.randn((num_gaussians, 3, 3)).cuda()
    Q, _ = torch.linalg.qr(A)  # Q ∈ O(3)

    # 4. Ensure Q ∈ SO(3): make det(Q) > 0
    det = torch.linalg.det(Q)  # shape: (N,)
    Q[det < 0] *= -1.0

    # 5. Compute Σ = Q D Qᵀ
    Sigma = Q @ D @ Q.transpose(1, 2)

    # 6. Add εI to improve numerical stability
    Sigma += epsilon * torch.eye(3).cuda().unsqueeze(0)

    # 7. Flatten to shape (N, 9)
    return Sigma.view(num_gaussians, 9)


# def splat_into_3d(grid_coords,
#                   means3d,
#                   opacities,
#                   covariances,
#                   vol_range,
#                   voxel_size,
#                   features=None,
#                   eps=1e-6):
#     grid_density = torch.zeros((*grid_coords.shape[:-1], 1), # (200, 200, 16, 1)
#                                device=grid_coords.device)
#     if features is not None:
#         grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
#                                  device=grid_coords.device)
#     all_lb = torch.zeros((num_gaussians, 3), dtype=torch.int32, device=grid_coords.device)
#     all_ub = torch.zeros((num_gaussians, 3), dtype=torch.int32, device=grid_coords.device)
#     all_maha_dist = [None] * means3d.size(0) # for debugging
#     for g in range(means3d.size(0)):
#         sigma = torch.sqrt(torch.diag(covariances[g])) # std for Gaussian
#         # create a factor with value 3 which will be multiplied with std, aiming at create a bound containing 99.7% Gaussian influence.
#         factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma)
#         bounds = means3d[g, None] + factor * sigma[None]
#         if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
#                 ((bounds < vol_range[None, 3:]).max(0).values.min())):
#             continue
#         bounds = bounds.clamp(vol_range[:3], vol_range[3:]) # only consider the influence of Gaussian within the boundary
#         bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist() # from absolute value to pixel coord
        
#         slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)]) # create a slice for each axis
        
#         # all_lb[g] = torch.tensor([s.start for s in slices], device=grid_coords.device, dtype=torch.int32)
#         # all_ub[g] = torch.tensor([s.stop-1 for s in slices], device=grid_coords.device, dtype=torch.int32)

#         diff = grid_coords[slices] - means3d[g] # offset of each voxel versus the Gaussian center
#         maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse()
#                      @ diff.unsqueeze(-1)).squeeze(-1) # Mahalanobis distance, same as GaussianFormer
#         # all_maha_dist[g] = maha_dist # for debugging
#         density = opacities[g] * torch.exp(-0.5 * maha_dist) # Gaussian density is computed with MH distance and opacity
#         grid_density[slices] += density # accumulate the density of each Gaussian
#         # voxel_count = density.numel()
#         # all_voxel_counts[g] = voxel_count
#         if features is not None:
#             grid_feats[slices] += density * features[g] # accumulate the feature of each Gaussian into voxel, the feature is weighted by density

#     if features is None:
#         # print("grid_density shape:", grid_density.shape)
#         # print("grid_density min:", grid_density.min())
#         # print("grid_density max:", grid_density.max())
#         # print("grid_density mean:", grid_density.mean())
#         # print("grid_density median:", grid_density.median())
#         # print("non empty voxels:", (grid_density > eps).sum().item())
#         return grid_density, all_lb, all_ub, all_maha_dist
#     grid_feats /= grid_density.clamp(eps)
#     return grid_density, grid_feats, all_lb, all_ub

def compare_gradients(name, grad_cuda, grad_sw, atol=1e-3, rtol=1e-3):
    diff = (grad_cuda - grad_sw).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    l2_diff = torch.norm(diff).item()
    max_idx = torch.argmax(diff)

    # Flatten for easy indexing
    grad_cuda_flat = grad_cuda.flatten()
    grad_sw_flat = grad_sw.flatten()

    print(f"\n{name} gradient comparison:")
    print(f"  Max diff: {max_diff}")
    print(f"  Mean diff: {mean_diff}")
    print(f"  L2 norm of diff: {l2_diff}")
    print(f"  Max diff index: {max_idx.item()}")
    # print(f"  CUDA grad at max diff: {grad_cuda_flat[max_idx].item():.6e}")
    # print(f"  SW   grad at max diff: {grad_sw_flat[max_idx].item():.6e}")

    if torch.allclose(grad_cuda, grad_sw, atol=atol, rtol=rtol):
        print("  ✅ Gradients match within tolerance.")
    else:
        print("  ❌ Gradients do NOT match.")


def print_topk_opacity_diff(opacity_cuda, opacity_sw, k=10):
    diff = (opacity_cuda - opacity_sw).abs()
    topk_vals, topk_idx = torch.topk(diff, k)
    print(f"\n🔍 Top-{k} largest differences in opacity gradients:")
    for i in range(k):
        idx = topk_idx[i].item()
        print(f"Index {idx}:")
        print(f"  CUDA: {opacity_cuda[idx].item()}")
        print(f"  SW  : {opacity_sw[idx].item()}")
        print(f"  Abs Diff: {topk_vals[i].item()}")

def print_topk_means_diff(means_cuda, means_sw, k=10):
    diff = (means_cuda - means_sw).abs()  # shape: [N, 3]
    diff_norm = diff.norm(p=1, dim=1)  # reduce to [N]
    k = min(k, diff_norm.numel())

    topk_vals, topk_idx = torch.topk(diff_norm, k)
    print(f"\n🔍 Top-{k} largest differences in mean gradients:")
    for i in range(k):
        idx = topk_idx[i].item()
        print(f"Index {idx}:")
        print(f"  CUDA: {means_cuda[idx].tolist()}")
        print(f"  SW  : {means_sw[idx].tolist()}")
        print(f"  Abs Diff (L1): {topk_vals[i].item()}")

def print_topk_covariance_diff(cov_cuda, cov_sw, k=10):
    """
    Compare CUDA and software gradients for covariance (flattened 3x3 -> shape [N, 9]).
    Prints top-k entries with the largest total absolute difference (L1 norm).
    """
    diff = (cov_cuda - cov_sw).abs()  # shape: [N, 9]
    diff_norm = diff.norm(p=1, dim=1)  # shape: [N]
    k = min(k, diff_norm.numel())

    topk_vals, topk_idx = torch.topk(diff_norm, k)
    print(f"\n🔍 Top-{k} largest differences in covariance gradients:")
    for i in range(k):
        idx = topk_idx[i].item()
        print(f"Index {idx}:")
        print(f"  CUDA: {cov_cuda[idx].tolist()}")
        print(f"  SW  : {cov_sw[idx].tolist()}")
        print(f"  Abs Diff (L1): {topk_vals[i].item()}")


if __name__ == "__main__":
    # torch.manual_seed(1024)

    num_gaussians = 10000
    voxel_size_list = [0.4, 0.4, 0.4]
    voxel_range_list = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

    means3D = torch.empty((num_gaussians, 3), device="cuda")
    means3D[:, 0] = torch.rand(num_gaussians, device="cuda") * 80.0 - 40.0
    means3D[:, 1] = torch.rand(num_gaussians, device="cuda") * 80.0 - 40.0
    means3D[:, 2] = torch.rand(num_gaussians, device="cuda") * 6.4 - 1.0

    opacity = torch.rand(num_gaussians, device="cuda")
    covariances = generate_random_covariances_with_rotation(num_gaussians)
    
    means3D_cuda = means3D.clone().detach().requires_grad_(True)
    opacity_cuda = opacity.clone().detach().requires_grad_(True)
    covariances_cuda = covariances.clone().detach().requires_grad_(True)

    voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size_list, voxel_range=voxel_range_list)
    torch.cuda.synchronize()
    density, grid_feats, stats = voxelizer(means3D_cuda.contiguous(), opacity_cuda.contiguous(), covariances_cuda.contiguous())
    torch.cuda.synchronize()
    print(f"CUDA forward time: {stats[0].item()} seconds")
    # density, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel = voxelizer(means3D.contiguous(), opacity.contiguous(), covariances.contiguous())
    density.retain_grad() 
    loss = density.sum()
    
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()
    print(f"CUDA backward time: {stats[1].item()} seconds")

    out_cuda = density.view(200, 200, 16, 1)
    # print("sum all density loss:", loss.item())
    # print("density grad shape:", density.grad.shape)
    # print("means3d cuda grad shape:", means3D_cuda.grad.shape)
    # print("opacity cuda grad shape:", opacity_cuda.grad.shape)
    # print("covariances cuda grad shape:", covariances_cuda.grad.shape)
    
    # print("valid_voxel_lb:", valid_voxel_lb)
    # print("valid_voxel_ub:", valid_voxel_ub)
    # print("num_valid_voxel:", num_valid_voxel)

    voxel_size = torch.tensor(voxel_size_list, device="cuda")
    vol_range = torch.tensor(voxel_range_list, device="cuda")
    grid_min, grid_max = vol_range[:3], vol_range[3:]
    grid_dim = ((grid_max - grid_min) / voxel_size).ceil().long()
    x = torch.arange(grid_dim[0], device="cuda") * voxel_size[0] + grid_min[0]
    y = torch.arange(grid_dim[1], device="cuda") * voxel_size[1] + grid_min[1]
    z = torch.arange(grid_dim[2], device="cuda") * voxel_size[2] + grid_min[2]
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    cov3x3 = covariances.view(num_gaussians, 3, 3)
    
    means3D_sw = means3D.clone().detach().requires_grad_(True)
    opacity_sw = opacity.clone().detach().requires_grad_(True)
    covariances_sw = cov3x3.clone().detach().requires_grad_(True)
    
    torch_gaussvoxel = GaussianVoxelizer(voxel_range_list, 0.4, filter_gaussians=False, opacity_thresh=0.6, covariance_thresh=1.5e-2)
    torch.cuda.synchronize()
    sw_start_time = time.time()
    # torch_grid_density, all_lb, all_ub, all_maha_dist = splat_into_3d(grid_coords, means3D_sw, opacity_sw, covariances_sw, vol_range, voxel_size)
    torch_grid_density = torch_gaussvoxel(means3D_sw.contiguous(), opacity_sw.contiguous(), covariances_sw.contiguous())
    sw_end_time = time.time()
    torch.cuda.synchronize()
    print(f"SW forward time: {sw_end_time - sw_start_time} seconds")
    torch_grid_density.retain_grad()
    loss_sw = torch_grid_density.sum()
    
    torch.cuda.synchronize()
    sw_backward_start_time = time.time()
    loss_sw.backward()
    torch.cuda.synchronize()
    sw_backward_end_time = time.time()
    print(f"SW backward time: {sw_backward_end_time - sw_backward_start_time} seconds")
    
    ####################### TAICHI ##########################
    torch.cuda.synchronize()
    taichi_start_time = time.time()
    # grid_density, grid_feats = taichi_splat_into_3d(means3D, opacity, cov3x3, vol_range, 0.4)
    taichi_density = taichi_splat_into_3d(means3D, opacity, cov3x3, vol_range, 0.4)
    taichi_end_time = time.time()
    torch.cuda.synchronize()
    print("TAICHI forward time:", taichi_end_time - taichi_start_time)
    loss_taichi = taichi_density.sum()

    # torch.cuda.synchronize()
    # taichi_backward_start_time = time.time()
    # loss_taichi.backward()
    # torch.cuda.synchronize()
    # taichi_backward_end_time = time.time()
    # print(f"TAICHI backward time: {taichi_backward_end_time - taichi_backward_start_time} seconds")

    print("sum all density loss (SW):", loss_sw.item())
    print("torch_grid_density grad shape:", torch_grid_density.grad.shape)
    # print("torch_grid_density content:", torch_grid_density.grad)

    compare_gradients("out gradient", density.grad, torch_grid_density.grad.squeeze(-1))

    # print("Grad (SW) - means3D shape:", means3D_sw.grad.shape)
    # print("Grad (SW) - opacity shape:", opacity_sw.grad.shape)
    # print("Grad (SW) - covariances:", covariances_sw.grad.shape)
    
    compare_gradients("Means3D", means3D_cuda.grad, means3D_sw.grad)
    compare_gradients("Opacity", opacity_cuda.grad, opacity_sw.grad)
    compare_gradients("Covariances", covariances_cuda.grad.view(-1, 3, 3), covariances_sw.grad)

    # print_topk_opacity_diff(opacity_cuda.grad, opacity_sw.grad, k=10)
    
    # print("opacity_sw dtype:", opacity_sw.dtype)
    # print("opacity_cuda.grad dtype:", opacity_cuda.grad.dtype)
    
    # print("Grad (CUDA) - means3D[0]:", means3D_cuda.grad[0])
    # print("Grad (SW) - means3D[0]:", means3D_sw.grad[0])
    # print("Grad (CUDA) - opacity[0]:", opacity_cuda.grad[0])
    # print("Grad (SW) - opacity[0]:", opacity_sw.grad[0])
    # print("Grad (CUDA) - covariances[0]:", covariances_cuda.grad[0])
    # print("Grad (SW) - covariances[0]:", covariances_sw.grad[0])
    
    # print_topk_means_diff(means3D_cuda.grad, means3D_sw.grad, k=10)
    
    # print_topk_covariance_diff(covariances_cuda.grad, covariances_sw.grad.view(-1, 9), k=10)