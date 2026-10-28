#include <torch/torch.h>
#include <cuda_gaussian_voxelizer.cuh>
#include <iostream>
#include <vector>

using namespace cudaprocess::gaussian_voxelizer;
using namespace torch::indexing;

void set_random_seeds(uint64_t seed = 42) {
    // 1. Set PyTorch CPU random seed
    torch::manual_seed(seed);
    
    // 2. Set PyTorch CUDA random seed (if CUDA is available)
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);  // For multi-GPU setups
    }
    
    // 3. Set C++ standard library random seed
    std::srand(static_cast<unsigned int>(seed));
    
    // 4. Set C random seed
    srand(static_cast<unsigned int>(seed));
    
    // 5. Optional: Set deterministic behavior for CUDA operations
    // Note: This may impact performance
    // torch::backends::cudnn::deterministic = true;
    // torch::backends::cudnn::benchmark = false;
    
    std::cout << "Random seeds set to: " << seed << std::endl;
}

torch::Tensor generate_random_covariances_with_rotation(int num_gaussians, float min_scale = 0.5, float max_scale = 8.0, float epsilon = 1e-4) {
    // 1. Sample random scaling (eigenvalues)
    torch::Tensor eigenvalues = torch::rand({num_gaussians, 3}, torch::kFloat32).to(torch::kCUDA);
    eigenvalues = eigenvalues * (max_scale - min_scale) + min_scale;

    // Create diagonal matrix D
    torch::Tensor D = torch::zeros({num_gaussians, 3, 3}, torch::kFloat32).to(torch::kCUDA);
    D.index_put_({Slice(), 0, 0}, eigenvalues.index({Slice(), 0}));
    D.index_put_({Slice(), 1, 1}, eigenvalues.index({Slice(), 1}));
    D.index_put_({Slice(), 2, 2}, eigenvalues.index({Slice(), 2}));

    // 2. Generate random orthogonal matrix R via QR decomposition
    torch::Tensor A = torch::randn({num_gaussians, 3, 3}, torch::kFloat32).to(torch::kCUDA);
    auto qr = torch::linalg_qr(A);
    torch::Tensor Q = std::get<0>(qr);  // Q ∈ SO(3) if adjusted

    // Optionally enforce det(Q) = +1 to ensure it's a rotation
    torch::Tensor det = torch::linalg_det(Q);
    torch::Tensor neg_mask = det.lt(0).view({-1, 1, 1});
    torch::Tensor Q_adj = Q.clone();
    Q_adj.index_put_({neg_mask.expand_as(Q)}, -Q.index({neg_mask.expand_as(Q)}));  // flip sign if needed

    // 3. Compute covariance: Σ = Q D Qᵀ
    torch::Tensor Sigma = torch::matmul(Q_adj, torch::matmul(D, Q_adj.transpose(1, 2)));

    // 4. (Optional) Add epsilon * I for numerical safety
    Sigma += torch::eye(3, torch::kFloat32).to(torch::kCUDA).unsqueeze(0) * epsilon;

    // 5. Flatten to shape (N, 9)
    return Sigma.view({num_gaussians, 9});
}

torch::Tensor invert_3x3_flat_torch(const torch::Tensor& cov, float eps = 1e-6f) {
    TORCH_CHECK(cov.sizes() == torch::IntArrayRef({3, 3}), "cov must be (3, 3)");

    // move to CPU
    auto cov_cpu = cov.cpu();
    const float* A = cov_cpu.contiguous().data_ptr<float>();

    float det = A[0]*(A[4]*A[8] - A[5]*A[7]) - A[1]*(A[3]*A[8] - A[5]*A[6]) + A[2]*(A[3]*A[7] - A[4]*A[6]);
    if (fabsf(det) < eps) det = eps;
    float inv_det = 1.0f / det;

    std::vector<float> Ainv(9);
    Ainv[0] =  (A[4]*A[8] - A[5]*A[7]) * inv_det;
    Ainv[1] = -(A[1]*A[8] - A[2]*A[7]) * inv_det;
    Ainv[2] =  (A[1]*A[5] - A[2]*A[4]) * inv_det;
    Ainv[3] = -(A[3]*A[8] - A[5]*A[6]) * inv_det;
    Ainv[4] =  (A[0]*A[8] - A[2]*A[6]) * inv_det;
    Ainv[5] = -(A[0]*A[5] - A[2]*A[3]) * inv_det;
    Ainv[6] =  (A[3]*A[7] - A[4]*A[6]) * inv_det;
    Ainv[7] = -(A[0]*A[7] - A[1]*A[6]) * inv_det;
    Ainv[8] =  (A[0]*A[4] - A[1]*A[3]) * inv_det;

    // recreate the tensor, then send it back to CUDA
    auto inv_tensor = torch::from_blob(Ainv.data(), {3, 3}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return inv_tensor.to(cov.device());
}




torch::Tensor splat_into_3d_cpp(
    const torch::Tensor& grid_coords,        // (X, Y, Z, 3)
    const torch::Tensor& means3d,            // (G, 3)
    const torch::Tensor& opacities,          // (G,)
    const torch::Tensor& covariances,        // (G, 3, 3)
    const std::vector<float>& vol_range,     // {xmin, ymin, zmin, xmax, ymax, zmax}
    const std::vector<float>& voxel_size,    // {vx, vy, vz}
    float eps = 1e-6
) {
    torch::Tensor grid_density = torch::zeros({grid_coords.size(0), grid_coords.size(1), grid_coords.size(2), 1}, grid_coords.options());
    auto vol_min = torch::tensor({vol_range[0], vol_range[1], vol_range[2]}, grid_coords.options());
    auto vol_max = torch::tensor({vol_range[3], vol_range[4], vol_range[5]}, grid_coords.options());
    auto voxel_sz = torch::tensor({voxel_size[0], voxel_size[1], voxel_size[2]}, grid_coords.options());

    int G = means3d.size(0);

    for (int g = 0; g < G; ++g) {
        auto mu = means3d[g];                              // (3,)
        auto cov = covariances[g];                         // (3, 3)
        // auto inv_cov = torch::linalg_inv(cov);             // (3, 3)
        auto inv_cov = invert_3x3_flat_torch(cov);
        auto sigma = torch::sqrt(cov.diag());              // (3,)
        auto factor = torch::tensor({-3.0f, 3.0f}, sigma.options()).view({2, 1});
        auto bounds = mu.view({1, 3}) + factor * sigma.view({1, 3});  // (2, 3)
        bounds = torch::max(bounds, vol_min.view({1, 3}));
        bounds = torch::min(bounds, vol_max.view({1, 3}));

        auto lo = ((bounds[0] - vol_min) / voxel_sz).floor().to(torch::kInt32);
        auto hi = ((bounds[1] - vol_min) / voxel_sz).ceil().to(torch::kInt32);
        if ((hi - lo).min().item<int>() <= 0) continue; // skip invalid region

        if (g < 5) {
            std::cout << "Gaussian " << g << ": bounds = " << bounds << ", lo = " << lo << ", hi = " << hi << std::endl;
        }

        auto slices = std::vector<torch::indexing::TensorIndex>{
            torch::indexing::Slice(lo[0].item<int>(), hi[0].item<int>() + 1),
            torch::indexing::Slice(lo[1].item<int>(), hi[1].item<int>() + 1),
            torch::indexing::Slice(lo[2].item<int>(), hi[2].item<int>() + 1)
        };

        // grid_coords[slices] shape: (x, y, z, 3)
        auto sub_grid = grid_coords.index(slices);                    // (..., 3)
        auto diff = sub_grid - mu;                                    // (..., 3)
        // printf("Splatting Gaussian %d: diff shape = %s\n", g, diff.sizes());
        // printf("diff")
        // compute the flattened voxel_idx of the sub-grid, matching the CUDA flattening rule
        auto x_range = torch::arange(lo[0].item<int>(), hi[0].item<int>() + 1, torch::kInt32).to(grid_coords.device());
        auto y_range = torch::arange(lo[1].item<int>(), hi[1].item<int>() + 1, torch::kInt32).to(grid_coords.device());
        auto z_range = torch::arange(lo[2].item<int>(), hi[2].item<int>() + 1, torch::kInt32).to(grid_coords.device());

        auto mesh = torch::meshgrid({x_range, y_range, z_range}, "ij");
        auto X = mesh[0];  // shape: (dx, dy, dz)
        auto Y = mesh[1];
        auto Z = mesh[2];

        // flatten the voxel index (CUDA layout)
        int dx = hi[0].item<int>() - lo[0].item<int>() + 1;
        int dy = hi[1].item<int>() - lo[1].item<int>() + 1;
        int dz = hi[2].item<int>() - lo[2].item<int>() + 1;

        int voxel_dim_x = grid_coords.size(0);
        int voxel_dim_y = grid_coords.size(1);
        int voxel_dim_z = grid_coords.size(2);

        auto voxel_idx = Z * (voxel_dim_y * voxel_dim_x) + Y * voxel_dim_x + X; // shape: (dx, dy, dz)
        voxel_idx = voxel_idx.reshape({-1});

        if (g < 1) {  // only look at the first Gaussian
            std::cout << "voxel_idx for Gaussian " << g << " (first 20):\n";
            auto voxel_idx_cpu = voxel_idx.to(torch::kCPU);
            auto acc = voxel_idx_cpu.accessor<int, 1>();
            for (int i = 0; i < std::min((long)20, voxel_idx_cpu.size(0)); i++) {
                std::cout << acc[i] << " ";
            }
            std::cout << std::endl;
        }


        auto diff_unsq = diff.unsqueeze(-2);                          // (..., 1, 3)
        auto maha = torch::matmul(torch::matmul(diff_unsq, inv_cov), diff_unsq.transpose(-1, -2)).squeeze(-1).squeeze(-1); // (...)
        auto density = opacities[g] * torch::exp(-0.5 * maha);        // (...)

        grid_density.index_put_(slices, grid_density.index(slices) + density.unsqueeze(-1));
    }

    return grid_density;
}



int main() {
    // set_random_seeds(12345);
    int num_gaussians = 1800;

    float x_min = -40.0f, x_max = 40.0f;
    float y_min = -40.0f, y_max = 40.0f;
    float z_min = -1.0f,  z_max = 5.4f;

    torch::Tensor means3D = torch::empty({num_gaussians, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    means3D.index({Slice(), 0}) = -torch::rand({num_gaussians}, torch::kCUDA) * 8 * (x_max - x_min) + x_min;
    means3D.index({Slice(), 1}) = -torch::rand({num_gaussians}, torch::kCUDA) * 8 * (y_max - y_min) + y_min;
    means3D.index({Slice(), 2}) = torch::rand({num_gaussians}, torch::kCUDA) * (z_max - z_min) + z_min;

    torch::Tensor opacity = torch::rand({num_gaussians}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    torch::Tensor covariances = generate_random_covariances_with_rotation(num_gaussians);

    std::vector<float> voxel_size_vec = {0.4f, 0.4f, 0.4f};
    std::vector<float> voxel_range_vec = {-40.f, -40.f, -1.f, 40.f, 40.f, 5.4f};

    // convert to CUDA tensors for the CUDA kernel
    auto voxel_size_tensor = torch::tensor({0.4f, 0.4f, 0.4f}, torch::kCUDA);
    auto voxel_range_tensor = torch::tensor({-40.f, -40.f, -1.f, 40.f, 40.f, 5.4f}, torch::kCUDA);

    // // ===== CUDA output =====
    // auto [density_tensor, cov_inv_tensor, valid_voxel_lb, valid_voxel_ub, num_voxels_per_gaussian, grid_feats, forward_time] =
    //     cudaprocess::gaussian_voxelizer::cuda_gaussian_voxelizer_forward(
    //         means3D, opacity, covariances, voxel_size_tensor, voxel_range_tensor);

    // std::vector<int64_t> dims = {200, 200, 16};
    // auto out_reshaped = density_tensor.view({dims[0], dims[1], dims[2], 1});
    // std::cout << "CUDA output shape: " << out_reshaped.sizes() << std::endl;

    // auto [grad_means3D, grad_cov, grad_opacity, grad_feats_grid, backward_time] =
    //     cudaprocess::gaussian_voxelizer::cuda_gaussian_voxelizer_backward(
    //         out_reshaped, means3D, opacity, cov_inv_tensor, voxel_size_tensor, voxel_range_tensor,
    //         valid_voxel_lb, valid_voxel_ub, num_voxels_per_gaussian, covariances);

    // ===== CUDA output =====
    // no features, pass nullopt here
    // c10::optional<torch::Tensor> no_features = c10::nullopt;
    torch::Tensor features = torch::rand({num_gaussians, 4},
        torch::device(torch::kCUDA).dtype(torch::kFloat32));
    c10::optional<torch::Tensor> opt_features = features;

    auto [density_tensor, cov_inv_tensor, valid_voxel_lb, valid_voxel_ub,
        num_voxels_per_gaussian, grid_feats, g2t_offsets, g2t_tile_ids, forward_time] =
        cudaprocess::gaussian_voxelizer::cuda_gaussian_voxelizer_forward(
            means3D, opacity, covariances, voxel_size_tensor, voxel_range_tensor, features);

    std::vector<int64_t> dims = {200, 200, 16};
    auto out_reshaped = density_tensor.view({dims[0], dims[1], dims[2], 1});
    std::cout << "CUDA output shape: " << out_reshaped.sizes() << std::endl;

    // ===== build the gradient inputs of the backward =====
    // grad_density_output: the shape must be [X,Y,Z], so use the shape of density_tensor directly
    // auto grad_density_output = torch::ones_like(density_tensor);
    // auto grad_density_output = density_tensor.view({dims[0], dims[1], dims[2]}).ones_like();
    auto grad_density_output = torch::ones({dims[0], dims[1], dims[2]}, density_tensor.options());



    // grad_feats_output: without features, pass a placeholder of shape [X,Y,Z,0]
    // auto grad_feats_output =
    //     torch::empty({dims[0], dims[1], dims[2], 0}, density_tensor.options());

    int Cf = features.size(1); // should be 4 here
    auto grad_feats_output = torch::zeros({dims[0], dims[1], dims[2], Cf}, density_tensor.options());

    

    // ===== call the backward =====
    auto [grad_means3D, grad_cov, grad_opacity, grad_feats_grid, backward_time] =
        cudaprocess::gaussian_voxelizer::cuda_gaussian_voxelizer_backward(
            /*grad_density_output*/ grad_density_output,
            /*grad_feats_output  */ grad_feats_output,
            /*means3D            */ means3D,
            /*opacity            */ opacity,
            /*cov_inv            */ cov_inv_tensor,
            /*voxel_size         */ voxel_size_tensor,
            /*voxel_range        */ voxel_range_tensor,
            /*valid_voxel_lb     */ valid_voxel_lb,
            /*valid_voxel_ub     */ valid_voxel_ub,
            /*num_vox_per_gauss  */ num_voxels_per_gaussian,
            /*g2t_offsets        */ g2t_offsets,
            /*g2t_tile_ids       */ g2t_tile_ids,
            /*density            */ density_tensor,
            /*grid_feats         */ grid_feats,
            /*features(opt)      */ features
        );

    std::cout << "Backward done. Time (s): " << backward_time << std::endl;


    // ===== CPU/C++ reference output =====
    auto x_vals = torch::arange(dims[0]) * voxel_size_vec[0] + voxel_range_vec[0];  // (X,)
    auto y_vals = torch::arange(dims[1]) * voxel_size_vec[1] + voxel_range_vec[1];  // (Y,)
    auto z_vals = torch::arange(dims[2]) * voxel_size_vec[2] + voxel_range_vec[2];  // (Z,)

    auto grid_x = x_vals.view({-1, 1, 1}).expand({dims[0], dims[1], dims[2]});
    auto grid_y = y_vals.view({1, -1, 1}).expand({dims[0], dims[1], dims[2]});
    auto grid_z = z_vals.view({1, 1, -1}).expand({dims[0], dims[1], dims[2]});

    torch::Tensor grid_coords = torch::stack({grid_x, grid_y, grid_z}, /*dim=*/-1).to(torch::kCUDA);
    grid_coords = grid_coords + 0.5;

    auto cov3x3 = covariances.view({num_gaussians, 3, 3});
    auto grid_density = splat_into_3d_cpp(grid_coords, means3D, opacity, cov3x3, voxel_range_vec, voxel_size_vec);
    std::cout << "CPU output shape: " << grid_density.sizes() << std::endl;

    // ===== comparison =====
    if (out_reshaped.sizes() != grid_density.sizes()) {
        std::cerr << "Mismatch in output shapes!" << std::endl;
        return 1;
    }

    auto diff = (out_reshaped - grid_density).abs();
    auto max_diff = diff.max().template item<float>();
    auto mean_diff = diff.mean().template item<float>();
    auto l2_error = torch::sqrt((diff * diff).sum()).template item<float>();

    std::cout << "\n===== Comparison =====" << std::endl;
    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Mean absolute difference: " << mean_diff << std::endl;
    std::cout << "L2 error: " << l2_error << std::endl;

    float tolerance = 1e-4;
    if (max_diff < tolerance) {
        std::cout << "✅ Outputs are numerically consistent within tolerance." << std::endl;
    } else {
        std::cout << "❌ Outputs differ significantly." << std::endl;
    }

    return 0;
}
