import torch
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer
from taichi_voxelizer import TaichiVoxelizer
from gaussian_voxelizer import GaussianVoxelizer
import time
def taichi_splat_into_3d(
    means3d,          # (G,3)
    opacities,        # (G,1) or (G,)
    covariances,      # (G,3,3)
    vol_range,        # (6,)
    voxel_size,       # float
    features=None,    # (G,C) or None
    eps=1e-6,
):
    # ---- host parameters ----
    vol_range_py = vol_range.detach().cpu().flatten().tolist() if torch.is_tensor(vol_range) else list(vol_range)
    voxel_size_f = float(voxel_size)

    # ---- shape and dtype fallback (2D/3D) ----
    means3d     = means3d.contiguous().to(torch.float32)          # (G,3)
    covariances = covariances.contiguous().to(torch.float32)      # (G,3,3)

    opacities = opacities.contiguous().to(torch.float32)          # (G,) or (G,1)
    if opacities.ndim == 1:
        opacities = opacities.unsqueeze(1)                        # -> (G,1)

    # want_feats = features is not None
    # if features is None:
    #     # Taichi initializes the field with features.n, so at least 1 dim is needed
    #     features = torch.ones((means3d.size(0), 1),
    #                           device=means3d.device, dtype=torch.float32)   # (G,1)
    # else:
    #     features = features.contiguous().to(torch.float32)                   # (G,C)
    
    want_feats = features is not None
    if features is None:
        # use placeholder features with at least 2 channels, the second channel is 0
        features = torch.zeros((means3d.size(0), 2),
                            device=means3d.device, dtype=torch.float32)
        features[:, 0] = 1.0
    else:
        features = features.contiguous().to(torch.float32)
        # if there is only 1 channel, pad to 2 channels (the second channel is 0)
        if features.shape[1] == 1:
            features = torch.cat([features, torch.zeros_like(features)], dim=1)


    # ---- add the batch dim (B=1) to fit @unbatched_forward ----
    means3d_b     = means3d.unsqueeze(0)          # (1,G,3)
    opacities_b   = opacities.unsqueeze(0)        # (1,G,1)
    covariances_b = covariances.unsqueeze(0)      # (1,G,3,3)
    features_b    = features.unsqueeze(0)         # (1,G,C)

    # ---- call TaichiVoxelizer (the class implementation is unchanged) ----
    vx = TaichiVoxelizer(
        vol_range=vol_range_py,
        voxel_size=voxel_size_f,
        filter_gaussians=False,
        opacity_thresh=0.0,
        eps=eps,
    )
    # after slicing by @unbatched_forward, __call__ receives (G,3)/(G,1)/(G,3,3)/(G,C), which satisfies the 2D/3D limit of tensor_to_field
    density_b, feats_b = vx(
        means3d=means3d_b,
        opacities=opacities_b,
        features=features_b,
        covariances=covariances_b,
    )
    # density_b: (1, Nx, Ny, Nz)
    # feats_b:   (1, Nx, Ny, Nz, C)

    density = density_b.squeeze(0).unsqueeze(-1)  # (Nx,Ny,Nz,1) to match CUDA
    if not want_feats:
        return density
    feats = feats_b.squeeze(0)                    # (Nx,Ny,Nz,C)
    return density, feats

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
        
#         all_lb[g] = torch.tensor([s.start for s in slices], device=grid_coords.device, dtype=torch.int32)
#         all_ub[g] = torch.tensor([s.stop-1 for s in slices], device=grid_coords.device, dtype=torch.int32)

#         diff = grid_coords[slices] - means3d[g] # offset of each voxel versus the Gaussian center
#         maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse()
#                      @ diff.unsqueeze(-1)).squeeze(-1) # Mahalanobis distance, same as GaussianFormer
#         all_maha_dist[g] = maha_dist # for debugging
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


if __name__ == "__main__":
    # torch.manual_seed(42)

    num_gaussians = 9000
    voxel_size_list = [0.4, 0.4, 0.4]
    voxel_range_list = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

    means3D = torch.empty((num_gaussians, 3), device="cuda")
    means3D[:, 0] = torch.rand(num_gaussians, device="cuda") * 80.0 - 40.0
    means3D[:, 1] = torch.rand(num_gaussians, device="cuda") * 80.0 - 40.0
    means3D[:, 2] = torch.rand(num_gaussians, device="cuda") * 6.4 - 1.0

    opacity = torch.rand(num_gaussians, device="cuda")
    covariances = generate_random_covariances_with_rotation(num_gaussians)

    voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size_list, voxel_range=voxel_range_list).to(means3D.device)
    # density, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel, d_maha_dist_0 = voxelizer(means3D.contiguous(), opacity.contiguous(), covariances.contiguous())
    density, grid_feats, stats = voxelizer(means3D.contiguous(), opacity.contiguous(), covariances.contiguous())

    print("cuda forward time:", stats[0].item())

    out_cuda = density.view(200, 200, 16, 1)

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
    torch_gaussvoxel = GaussianVoxelizer(voxel_range_list, 0.4, filter_gaussians=False, opacity_thresh=0.6, covariance_thresh=1.5e-2)
    sw_start_time = time.time()
    torch_grid_density = torch_gaussvoxel(means3D.contiguous(), opacity.contiguous(), cov3x3.contiguous())
    sw_end_time = time.time()
    print("splat_into_3d time:", sw_end_time - sw_start_time)

    ####################### TAICHI ##########################
    taichi_start_time = time.time()
    # grid_density, grid_feats = taichi_splat_into_3d(means3D, opacity, cov3x3, vol_range, 0.4)
    taichi_density = taichi_splat_into_3d(means3D, opacity, cov3x3, vol_range, 0.4)

    taichi_end_time = time.time()
    print("taichi_splat_into_3d time:", taichi_end_time - taichi_start_time)

    # print("all_lb:", all_lb)
    # print("all_ub:", all_ub)
    # print("all_maha_dist[0]:", all_maha_dist[0])
    # print("all_maha_dist[0] shape:", all_maha_dist[0].shape)
    
    # print("\n===== boundary Comparison =====")
    # lb_diff = (valid_voxel_lb - all_lb).abs()
    # ub_diff = (valid_voxel_ub - all_ub).abs()

    # print("LB max diff:", lb_diff.max().item())
    # print("UB max diff:", ub_diff.max().item())

    # if lb_diff.max().item() == 0 and ub_diff.max().item() == 0:
    #     print("✅ LB/UB match exactly")
    # else:
    #     print("❌ LB/UB differ")
    #     diff_indices = torch.nonzero((lb_diff.sum(dim=1) > 0) | (ub_diff.sum(dim=1) > 0)).squeeze(1)
    #     for idx in diff_indices.tolist():
    #         print(f"\nGaussian {idx}:")
    #         print(f"  CUDA LB: {valid_voxel_lb[idx].tolist()}  |  Python LB: {all_lb[idx].tolist()}")
    #         print(f"  CUDA UB: {valid_voxel_ub[idx].tolist()}  |  Python UB: {all_ub[idx].tolist()}")
        
    # print("\n===== Gaussian 0 maha dis Comparison =====")
    # print("d_maha_dist_0 shape:", d_maha_dist_0.shape)
    # print("all_maha_dist[0] shape:", all_maha_dist[0].squeeze(-1).shape)
    # maha_diff = (d_maha_dist_0 - all_maha_dist[0].squeeze(-1)).abs()
    # max_maha_diff = maha_diff.max().item()
    # mean_maha_diff = maha_diff.mean().item()
    # l2_maha_error = torch.sqrt((maha_diff * maha_diff).sum()).item()
    # print("Max maha diff:", max_maha_diff)
    # print("Mean maha diff:", mean_maha_diff)
    # print("L2 maha error:", l2_maha_error)

    diff = (out_cuda - torch_grid_density).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    l2_error = torch.sqrt((diff * diff).sum()).item()
    
    diff_taichi = (out_cuda - taichi_density)
    taichi_max_diff = diff_taichi.max().item()
    taichi_mean_diff = diff_taichi.mean().item()
    taichi_l2_error = torch.sqrt((diff_taichi * diff_taichi).sum()).item()

    print(f"Output shape (CUDA): {out_cuda.shape}")
    print(f"Output shape (SW):   {torch_grid_density.shape}")
    print("\n===== Comparison with sw =====")
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"L2 error: {l2_error}")
    
    tol = 1e-4
    if max_diff < tol:
        print("✅ Outputs are numerically consistent within tolerance.")
    else:
        print("❌ Outputs differ significantly.")

    print("\n===== Comparison with Taichi =====")
    print(f"Max absolute difference: {taichi_max_diff}")
    print(f"Mean absolute difference: {taichi_mean_diff}")
    print(f"L2 error: {taichi_l2_error}")

    tol = 1e-4
    if taichi_max_diff < tol:
        print("✅ Outputs are numerically consistent within tolerance.")
    else:
        print("❌ Outputs differ significantly.")