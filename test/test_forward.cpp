#include <torch/torch.h>
#include <cuda_gaussian_voxelizer.cuh>
#include <iostream>
#include <vector>

using namespace cudaprocess::gaussian_voxelizer;
using namespace torch::indexing;

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

        auto slices = std::vector<torch::indexing::TensorIndex>{
            torch::indexing::Slice(lo[0].item<int>(), hi[0].item<int>() + 1),
            torch::indexing::Slice(lo[1].item<int>(), hi[1].item<int>() + 1),
            torch::indexing::Slice(lo[2].item<int>(), hi[2].item<int>() + 1)
        };

        // grid_coords[slices] shape: (x, y, z, 3)
        auto sub_grid = grid_coords.index(slices);                    // (..., 3)
        auto diff = sub_grid - mu;                                    // (..., 3)
        auto diff_unsq = diff.unsqueeze(-2);                          // (..., 1, 3)
        auto maha = torch::matmul(torch::matmul(diff_unsq, inv_cov), diff_unsq.transpose(-1, -2)).squeeze(-1).squeeze(-1); // (...)
        auto density = opacities[g] * torch::exp(-0.5 * maha);        // (...)

        grid_density.index_put_(slices, grid_density.index(slices) + density.unsqueeze(-1));
    }

    return grid_density;
}




int main() {
    int num_gaussians = 1800;

    float x_min = -40.0f, x_max = 40.0f;
    float y_min = -40.0f, y_max = 40.0f;
    float z_min = -1.0f,  z_max = 5.4f;

    torch::Tensor means3D = torch::empty({num_gaussians, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    means3D.index({Slice(), 0}) = torch::rand({num_gaussians}, torch::kCUDA) * (x_max - x_min) + x_min;
    means3D.index({Slice(), 1}) = torch::rand({num_gaussians}, torch::kCUDA) * (y_max - y_min) + y_min;
    means3D.index({Slice(), 2}) = torch::rand({num_gaussians}, torch::kCUDA) * (z_max - z_min) + z_min;

    torch::Tensor opacity = torch::rand({num_gaussians}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    // torch::Tensor covariances = torch::rand({num_gaussians, 9}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    torch::Tensor covariances = generate_random_covariances_with_rotation(num_gaussians);

    std::vector<float> voxel_size = {0.4f, 0.4f, 0.4f};
    std::vector<float> voxel_range = {-40.f, -40.f, -1.f, 40.f, 40.f, 5.4f};

    CudaGaussianVoxelizer voxelizer(
        voxel_range.data(),
        voxel_size.data(),
        0.6f,
        1.5e-2f,
        num_gaussians,
        true
    );
    std::vector<int64_t> dims = {200, 200, 16};
    auto out = voxelizer.forward(means3D, opacity, covariances);
    auto out_reshaped = out.view({dims[0], dims[1], dims[2], 1});
    std::cout << "Output shape: " << out_reshaped.sizes() << std::endl;

    // std::cout<< "Output contents: " << out_reshaped << std::endl;

//     // build grid_coords
    
//     torch::Tensor grid_coords = torch::empty({dims[0], dims[1], dims[2], 3}, torch::kFloat32).to(torch::kCUDA);
//     for (int i = 0; i < dims[0]; ++i) {
//         for (int j = 0; j < dims[1]; ++j) {
//             for (int k = 0; k < dims[2]; ++k) {
//                 float x = voxel_range[0] + i * voxel_size[0];
//                 float y = voxel_range[1] + j * voxel_size[1];
//                 float z = voxel_range[2] + k * voxel_size[2];
//                 grid_coords[i][j][k][0] = x;
//                 grid_coords[i][j][k][1] = y;
//                 grid_coords[i][j][k][2] = z;
//             }
//         }
//     }

//     auto cov3x3 = covariances.view({num_gaussians, 3, 3});
//     auto grid_density = splat_into_3d_cpp(grid_coords, means3D, opacity, cov3x3, voxel_range, voxel_size);
//     std::cout << "Grid density shape: " << grid_density.sizes() << std::endl;
//     std::cout << "Grid density contents: " << grid_density << std::endl;

//     // compare_results(out, grid_density.view({dims[0], dims[1], dims[2], 1}), 1e-5, "CUDA vs C++");
//     // 1. make sure the output shapes match
//     if (out_reshaped.sizes() != grid_density.sizes()) {
//         std::cerr << "Mismatch in output shapes!" << std::endl;
//         return 1;
//     }

//     // 2. compute the error metrics (L1/L2/max difference)
//     auto diff = (out_reshaped - grid_density).abs();
//     auto max_diff = diff.max().item<float>();
//     auto mean_diff = diff.mean().item<float>();
//     auto l2_error = torch::sqrt((diff * diff).sum()).item<float>();

//     std::cout << "\n===== Comparison =====" << std::endl;
//     std::cout << "Max absolute difference: " << max_diff << std::endl;
//     std::cout << "Mean absolute difference: " << mean_diff << std::endl;
//     std::cout << "L2 error: " << l2_error << std::endl;

//     // 3. check whether the outputs match (within tolerance)
//     float tolerance = 1e-4;
//     if (max_diff < tolerance) {
//         std::cout << "✅ Outputs are numerically consistent within tolerance." << std::endl;
//     } else {
//         std::cout << "❌ Outputs differ significantly." << std::endl;
//     }


    return 0;
}
