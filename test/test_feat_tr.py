import torch
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer
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


def splat_into_3d(grid_coords,
                  means3d,
                  opacities,
                  covariances,
                  vol_range,
                  voxel_size,
                  features=None,
                  eps=1e-6):
    grid_density = torch.zeros((*grid_coords.shape[:-1], 1), # (200, 200, 16, 1)
                               device=grid_coords.device)
    if features is not None:
        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                 device=grid_coords.device)
    # all_lb = torch.zeros((num_gaussians, 3), dtype=torch.int32, device=grid_coords.device)
    # all_ub = torch.zeros((num_gaussians, 3), dtype=torch.int32, device=grid_coords.device)
    # all_maha_dist = [None] * means3d.size(0) # for debugging
    for g in range(means3d.size(0)):
        sigma = torch.sqrt(torch.diag(covariances[g])) # std for Gaussian
        # create a factor with value 3 which will be multiplied with std, aiming at create a bound containing 99.7% Gaussian influence.
        factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma)
        bounds = means3d[g, None] + factor * sigma[None]
        if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                ((bounds < vol_range[None, 3:]).max(0).values.min())):
            continue
        bounds = bounds.clamp(vol_range[:3], vol_range[3:]) # only consider the influence of Gaussian within the boundary
        bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist() # from absolute value to pixel coord
        
        slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)]) # create a slice for each axis
        
        # all_lb[g] = torch.tensor([s.start for s in slices], device=grid_coords.device, dtype=torch.int32)
        # all_ub[g] = torch.tensor([s.stop-1 for s in slices], device=grid_coords.device, dtype=torch.int32)

        diff = grid_coords[slices] - means3d[g] # offset of each voxel versus the Gaussian center
        maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse()
                     @ diff.unsqueeze(-1)).squeeze(-1) # Mahalanobis distance, same as GaussianFormer
        # all_maha_dist[g] = maha_dist # for debugging
        density = opacities[g] * torch.exp(-0.5 * maha_dist) # Gaussian density is computed with MH distance and opacity
        grid_density[slices] += density # accumulate the density of each Gaussian
        # voxel_count = density.numel()
        # all_voxel_counts[g] = voxel_count
        if features is not None:
            grid_feats[slices] += density * features[g] # accumulate the feature of each Gaussian into voxel, the feature is weighted by density

    if features is None:
        # print("grid_density shape:", grid_density.shape)
        # print("grid_density min:", grid_density.min())
        # print("grid_density max:", grid_density.max())
        # print("grid_density mean:", grid_density.mean())
        # print("grid_density median:", grid_density.median())
        # print("non empty voxels:", (grid_density > eps).sum().item())
        return grid_density, None
    grid_feats /= grid_density.clamp(eps)
    return grid_density, grid_feats

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

if __name__ == "__main__":
    torch.manual_seed(42)

    num_gaussians = 1800
    voxel_size_list = [0.4, 0.4, 0.4]
    voxel_range_list = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

    means3D = torch.empty((num_gaussians, 3), device="cuda")
    means3D[:, 0] = torch.rand(num_gaussians, device="cuda") * 80.0 - 40.0
    means3D[:, 1] = torch.rand(num_gaussians, device="cuda") * 80.0 - 40.0
    means3D[:, 2] = torch.rand(num_gaussians, device="cuda") * 6.4 - 1.0

    opacity = torch.rand(num_gaussians, device="cuda")
    covariances = generate_random_covariances_with_rotation(num_gaussians)
    
    num_feats_dim = 12  # Example feature dimension
    features = torch.randn(num_gaussians, num_feats_dim, device="cuda")  # Example features, shape (N, F)
    
    # means3D = means3D.require_grad_(True)
    # opacity = opacity.require_grad_(True)
    # covariances = covariances.require_grad_(True)
    # features = features.require_grad_(True)
    means3D_cuda = means3D.clone().detach().requires_grad_(True)
    opacity_cuda = opacity.clone().detach().requires_grad_(True)
    covariances_cuda = covariances.clone().detach().requires_grad_(True)
    features = features.clone().detach().requires_grad_(True)

    # voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size_list, voxel_range=voxel_range_list)
    # # density, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel, d_maha_dist_0 = voxelizer(means3D.contiguous(), opacity.contiguous(), covariances.contiguous())
    # density, grid_feats_cuda = voxelizer(means3D.contiguous(), opacity.contiguous(), covariances.contiguous(), features.contiguous())
    
    # cuda_loss = density.sum() + grid_feats_cuda.sum()
    # cuda_loss.backward()
    
    # out_cuda = density.view(200, 200, 16, 1)

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
    
    means3D_tr = means3D.clone().detach().requires_grad_(True)
    opacity_tr = opacity.clone().detach().requires_grad_(True)
    covariances_tr = cov3x3.clone().detach().requires_grad_(True)
    features_tr = features.clone().detach().requires_grad_(True)
    out_tr, grid_feats = splat_into_3d(grid_coords, means3D_tr, opacity_tr, covariances_tr, vol_range, voxel_size, features=features_tr)
    out_tr.retain_grad()
    grid_feats.retain_grad()
    loss_tr = out_tr.sum() + grid_feats.sum()
    loss_tr.backward()
    
    print("out_tr grad shape:", out_tr.grad.shape)
    print("grid_feats grad shape:", grid_feats.grad.shape)
    
    
    means3D_tr1 = means3D.clone().detach().requires_grad_(True)
    opacity_tr1 = opacity.clone().detach().requires_grad_(True)
    covariances_tr1 = cov3x3.clone().detach().requires_grad_(True)
    features_tr1 = features.clone().detach().requires_grad_(True)

    out_tr1, grid_feats1 = splat_into_3d(grid_coords, means3D_tr1, opacity_tr1, covariances_tr1, vol_range, voxel_size, features=features_tr1)

    # gradient of out_tr only
    loss1 = out_tr1.sum()
    loss1.backward()
    grad_from_out_tr = means3D_tr1.grad.clone()

    # reset and take the gradient of grid_feats only
    means3D_tr2 = means3D.clone().detach().requires_grad_(True)
    opacity_tr2 = opacity.clone().detach().requires_grad_(True)
    covariances_tr2 = cov3x3.clone().detach().requires_grad_(True)
    features_tr2 = features.clone().detach().requires_grad_(True)

    out_tr2, grid_feats2 = splat_into_3d(grid_coords, means3D_tr2, opacity_tr2, covariances_tr2, vol_range, voxel_size, features=features_tr2)

    loss2 = grid_feats2.sum() + out_tr2.sum()
    loss2.backward()
    grad_from_grid_feats = means3D_tr2.grad.clone()

    # check: the total gradient should equal the sum of the two parts
    print("norm of the total gradient:", torch.norm(means3D_tr.grad).item())
    print("norm of the gradient from out_tr:", torch.norm(grad_from_out_tr).item())
    print("norm of the gradient from grid_feats:", torch.norm(grad_from_grid_feats).item())
    print("norm of the sum of the two parts:", torch.norm(grad_from_out_tr + grad_from_grid_feats).item())

    # check whether they are equal
    print("gradients match:", torch.allclose(means3D_tr.grad, grad_from_out_tr + grad_from_grid_feats, atol=1e-3))

    # diff = (out_cuda - out_tr).abs()
    # max_diff = diff.max().item()
    # mean_diff = diff.mean().item()
    # l2_error = torch.sqrt((diff * diff).sum()).item()

    # print(f"Output shape (CUDA): {out_cuda.shape}")
    # print(f"Output shape (SW):   {out_tr.shape}")
    # print("\n===== Comparison =====")
    # print(f"Max absolute difference: {max_diff}")
    # print(f"Mean absolute difference: {mean_diff}")
    # print(f"L2 error: {l2_error}")
    
    # tol = 1e-4
    # if max_diff < tol:
    #     print("✅ Outputs are numerically consistent within tolerance.")
    # else:
    #     print("❌ Outputs differ significantly.")
    
    # print("grid_feats_cuda shape:", grid_feats_cuda.shape)
    # print("grid_feats shape:", grid_feats.shape)
    
    # diff = (grid_feats_cuda - grid_feats).abs()
    # max_diff = diff.max().item()
    # mean_diff = diff.mean().item()
    # l2_error = torch.sqrt((diff * diff).sum()).item()
    # print("\n===== Grid feature Comparison =====")
    # print(f"Max absolute difference: {max_diff}")
    # print(f"Mean absolute difference: {mean_diff}")
    # print(f"L2 error: {l2_error}")
    
    # tol = 1e-4
    # if max_diff < tol:
    #     print("✅ Grid Feature Outputs are numerically consistent within tolerance.")
    # else:
    #     print("❌ Grid Feature Outputs differ significantly.")
        
        
    ####################### Backward check #######################
    # print("\n===== Backward Check =====")
    
    # compare_gradients("out gradient", density.grad, out_tr.grad.squeeze(-1))
    
    # compare_gradients("Means3D", means3D_cuda.grad, means3D_tr.grad)
    # compare_gradients("Opacity", opacity_cuda.grad, opacity_tr.grad)
    # compare_gradients("Covariances", covariances_cuda.grad.view(-1, 3, 3), covariances_tr.grad)
    