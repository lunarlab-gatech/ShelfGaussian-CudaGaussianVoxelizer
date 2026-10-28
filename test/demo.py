import torch
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer, PyCudaFeatsCosineLoss

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

    # 5. Compute covariance matrices
    Sigma = Q @ D @ Q.transpose(1, 2)

    # 6. Add εI to improve numerical stability
    Sigma += epsilon * torch.eye(3).cuda().unsqueeze(0)

    # 7. Flatten to shape (N, 9)
    return Sigma.view(num_gaussians, 9)

num_gaussians = 18000
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


# CUDA voxelizer
voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size_list, voxel_range=voxel_range_list)

density, _, _ = voxelizer(means3D_cuda.contiguous(), opacity_cuda.contiguous(), covariances_cuda.contiguous())

# ======================== CUDA feats loss ========================
X = int(round((voxel_range_list[3] - voxel_range_list[0]) / voxel_size_list[0]))  # 200
Y = int(round((voxel_range_list[4] - voxel_range_list[1]) / voxel_size_list[1]))  # 200
Z = int(round((voxel_range_list[5] - voxel_range_list[2]) / voxel_size_list[2]))  # 16
C = 512  # keep moderate to avoid OOM; raise if you have VRAM

# ----- Random predicted features -----
grid_feats = torch.randn(X, Y, Z, C, device="cuda", dtype=torch.float32)

# ----- Build masks -----
p = 0.25
voxels_label_full   = (torch.rand(X, Y, Z, device="cuda") < p)        # [X,Y,Z] bool
occ_gt_mask_camera  = (torch.rand(1, X, Y, Z, device="cuda") < 0.7)   # [1,X,Y,Z] bool
    
loss_calculator = PyCudaFeatsCosineLoss(voxel_size=voxel_size_list, voxel_range=voxel_range_list).to("cuda")
pseudo_dino_gt = -grid_feats.clone().detach().unsqueeze(0).contiguous()
loss_cuda, _ = loss_calculator(
    voxels_label_full.float(),            # [X,Y,Z] float
    occ_gt_mask_camera.squeeze(0),        # [X,Y,Z] bool
    grid_feats,                            # [X,Y,Z,C] float32
    pseudo_dino_gt.squeeze(0)              # [X,Y,Z,C] float32
)
print("Loss (GT = -pred):", float(loss_cuda.item()))