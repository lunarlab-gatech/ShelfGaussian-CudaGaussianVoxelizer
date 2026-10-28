// test_eval_mm.cpp
#include <torch/torch.h>
#include <torch/serialize.h>
#include <cuda_gaussian_voxelizer.cuh>

#include <c10/cuda/CUDACachingAllocator.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <cmath>

using namespace cudaprocess::gaussian_voxelizer;
using namespace torch::indexing;
namespace fs = std::filesystem;

// ---------------------- helper functions/declarations ----------------------
void set_random_seeds(uint64_t seed = 42) {
    torch::manual_seed(seed);
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);
    }
    std::srand(static_cast<unsigned int>(seed));
    srand(static_cast<unsigned int>(seed));
    std::cout << "Random seeds set to: " << seed << std::endl;
}

// forward declaration: CPU reference implementation
torch::Tensor splat_into_3d_cpp(
    const torch::Tensor& grid_coords,        // (X, Y, Z, 3)
    const torch::Tensor& means3d,            // (G, 3)
    const torch::Tensor& opacities,          // (G,)
    const torch::Tensor& covariances,        // (G, 3, 3)
    const std::vector<float>& vol_range,     // {xmin, ymin, zmin, xmax, ymax, zmax}
    const std::vector<float>& voxel_size,    // {vx, vy, vz}
    float eps = 1e-6f
);

// hand-written 3x3 matrix inverse (avoids version differences)
torch::Tensor invert_3x3_flat_torch(const torch::Tensor& cov, float eps = 1e-6f) {
    TORCH_CHECK(cov.sizes() == torch::IntArrayRef({3, 3}), "cov must be (3, 3)");
    auto cov_cpu = cov.cpu().contiguous();
    const float* A = cov_cpu.data_ptr<float>();

    float det = A[0]*(A[4]*A[8] - A[5]*A[7]) - A[1]*(A[3]*A[8] - A[5]*A[6]) + A[2]*(A[3]*A[7] - A[4]*A[6]);
    if (std::fabs(det) < eps) det = (det >= 0 ? eps : -eps);
    float inv_det = 1.0f / det;

    float B[9];
    B[0] =  (A[4]*A[8] - A[5]*A[7]) * inv_det;
    B[1] = -(A[1]*A[8] - A[2]*A[7]) * inv_det;
    B[2] =  (A[1]*A[5] - A[2]*A[4]) * inv_det;
    B[3] = -(A[3]*A[8] - A[5]*A[6]) * inv_det;
    B[4] =  (A[0]*A[8] - A[2]*A[6]) * inv_det;
    B[5] = -(A[0]*A[5] - A[2]*A[3]) * inv_det;
    B[6] =  (A[3]*A[7] - A[4]*A[6]) * inv_det;
    B[7] = -(A[0]*A[7] - A[1]*A[6]) * inv_det;
    B[8] =  (A[0]*A[4] - A[1]*A[3]) * inv_det;

    auto inv_tensor = torch::from_blob(B, {3, 3}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return inv_tensor.to(cov.device());
}

// CPU reference implementation: splat G Gaussians into the voxel grid
torch::Tensor splat_into_3d_cpp(
    const torch::Tensor& grid_coords,        // (X, Y, Z, 3)
    const torch::Tensor& means3d,            // (G, 3)
    const torch::Tensor& opacities,          // (G,)
    const torch::Tensor& covariances,        // (G, 3, 3)
    const std::vector<float>& vol_range,     // {xmin, ymin, zmin, xmax, ymax, zmax}
    const std::vector<float>& voxel_size,    // {vx, vy, vz}
    float eps
) {
    TORCH_CHECK(grid_coords.dim() == 4 && grid_coords.size(-1) == 3, "grid_coords must be (X,Y,Z,3)");
    TORCH_CHECK(means3d.size(1) == 3, "means3d must be (G,3)");
    TORCH_CHECK(covariances.sizes() == torch::IntArrayRef({means3d.size(0), 3, 3}), "covariances must be (G,3,3)");
    TORCH_CHECK(opacities.size(0) == means3d.size(0), "opacities must be (G,)");

    auto opts = grid_coords.options();
    auto grid_density = torch::zeros({grid_coords.size(0), grid_coords.size(1), grid_coords.size(2), 1}, opts);

    auto vol_min = torch::tensor({vol_range[0], vol_range[1], vol_range[2]}, opts);
    auto vol_max = torch::tensor({vol_range[3], vol_range[4], vol_range[5]}, opts);
    auto voxel_sz = torch::tensor({voxel_size[0], voxel_size[1], voxel_size[2]}, opts);

    const int G = static_cast<int>(means3d.size(0));

    for (int g = 0; g < G; ++g) {
        auto mu  = means3d[g];          // (3,)
        auto cov = covariances[g];      // (3,3)
        auto inv_cov = invert_3x3_flat_torch(cov);

        auto sigma  = torch::sqrt(cov.diag()); // (3,)
        auto factor = torch::tensor({-3.0f, 3.0f}, sigma.options()).view({2, 1});
        auto bounds = mu.view({1, 3}) + factor * sigma.view({1, 3});  // (2,3)
        bounds = torch::max(bounds, vol_min.view({1, 3}));
        bounds = torch::min(bounds, vol_max.view({1, 3}));

        auto lo = ((bounds[0] - vol_min) / voxel_sz).floor().to(torch::kInt32);
        auto hi = ((bounds[1] - vol_min) / voxel_sz).ceil().to(torch::kInt32);
        if ((hi - lo).min().item<int>() <= 0) continue;

        std::vector<torch::indexing::TensorIndex> slices{
            torch::indexing::Slice(lo[0].item<int>(), hi[0].item<int>() + 1),
            torch::indexing::Slice(lo[1].item<int>(), hi[1].item<int>() + 1),
            torch::indexing::Slice(lo[2].item<int>(), hi[2].item<int>() + 1)
        };

        auto sub_grid = grid_coords.index(slices); // (...,3)
        auto diff = sub_grid - mu;                 // (...,3)
        auto diff_unsq = diff.unsqueeze(-2);       // (...,1,3)
        auto maha = torch::matmul(torch::matmul(diff_unsq, inv_cov), diff_unsq.transpose(-1, -2))
                        .squeeze(-1).squeeze(-1);  // (...)
        auto density = opacities[g] * torch::exp(-0.5f * maha); // (...)

        grid_density.index_put_(slices, grid_density.index(slices) + density.unsqueeze(-1));
    }

    return grid_density;
}

// load .pt/.pth: must be a dict saved by torch.save with keys means3d/opacities/covariances/features
struct VoxInputs {
    torch::Tensor means3d;
    torch::Tensor opacities;
    torch::Tensor covariances; // (N,9) or (N,3,3)
    torch::Tensor features;    // (N,F)
};

// VoxInputs load_four_tensors_from_pth(const std::string& path, const c10::Device& device) {
//     std::ifstream in(path, std::ios::binary);
//     if (!in) throw std::runtime_error("cannot open: " + path);

//     std::vector<char> buffer((std::istreambuf_iterator<char>(in)),
//                               std::istreambuf_iterator<char>());

//     c10::IValue iv = torch::pickle_load(buffer);
//     auto dict = iv.toGenericDict();

//     auto get = [&](const char* key)->torch::Tensor {
//         auto it = dict.find(c10::IValue(std::string(key)));
//         if (it == dict.end()) throw std::runtime_error(std::string("key missing: ") + key);
//         return it->value().toTensor().to(device).contiguous().to(torch::kFloat32);
//     };

//     VoxInputs out;
//     out.means3d     = get("means3d");     // (N,3)
//     out.opacities   = get("opacities");   // (N,)
//     out.covariances = get("covariances"); // (N,9) or (N,3,3)
//     out.features    = get("features");    // (N,F)
//     return out;
// }

VoxInputs load_four_tensors_from_pth(const std::string& path) {
    c10::Device dev0(torch::kCUDA, 0);            // target device: cuda:0
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + path);

    std::vector<char> buffer((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());

    c10::IValue iv = torch::pickle_load(buffer);
    auto dict = iv.toGenericDict();

    auto get = [&](const char* key)->torch::Tensor {
        auto it = dict.find(c10::IValue(std::string(key)));
        if (it == dict.end()) throw std::runtime_error(std::string("key missing: ") + key);
        // move to CPU first (drops the original device binding), then to cuda:0
        return it->value().toTensor()
                 .to(torch::kCPU)                // CPU first
                 .contiguous()
                 .to(dev0)                       // then cuda:0
                 .to(torch::kFloat32);           // unify the precision if needed
    };

    VoxInputs out;
    out.means3d     = get("means3d");     // (N,3)
    out.opacities   = get("opacities");   // (N,)
    out.covariances = get("covariances"); // (N,9) or (N,3,3)
    out.features    = get("features");    // (N,F)
    return out;
}


// ---------------------- main program ----------------------
int main() {
    // set_random_seeds(12345);

    // config
    std::string data_folder    = "../debug_gaussian2voxel";
    std::vector<float> voxel_size_vec  = {0.4f, 0.4f, 0.4f};
    std::vector<float> voxel_range_vec = {-40.f, -40.f, -1.f, 40.f, 40.f, 5.4f};

    c10::Device device(torch::kCUDA);
    auto voxel_size_tensor  = torch::tensor({voxel_size_vec[0], voxel_size_vec[1], voxel_size_vec[2]}).to(device);
    auto voxel_range_tensor = torch::tensor({voxel_range_vec[0], voxel_range_vec[1], voxel_range_vec[2],
                                             voxel_range_vec[3], voxel_range_vec[4], voxel_range_vec[5]}).to(device);

    // compute the voxel dims from the range and the voxel size
    const int X = static_cast<int>(std::ceil((voxel_range_vec[3]-voxel_range_vec[0]) / voxel_size_vec[0]));
    const int Y = static_cast<int>(std::ceil((voxel_range_vec[4]-voxel_range_vec[1]) / voxel_size_vec[1]));
    const int Z = static_cast<int>(std::ceil((voxel_range_vec[5]-voxel_range_vec[2]) / voxel_size_vec[2]));
    std::vector<int64_t> dims = {X, Y, Z};

    // build the grid coordinates (X,Y,Z,3) on CUDA
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto x_vals = torch::arange(static_cast<float>(X), opts_f) * voxel_size_vec[0] + voxel_range_vec[0];
    auto y_vals = torch::arange(static_cast<float>(Y), opts_f) * voxel_size_vec[1] + voxel_range_vec[1];
    auto z_vals = torch::arange(static_cast<float>(Z), opts_f) * voxel_size_vec[2] + voxel_range_vec[2];
    auto grid_x = x_vals.view({-1, 1, 1}).expand({X, Y, Z});
    auto grid_y = y_vals.view({ 1,-1, 1}).expand({X, Y, Z});
    auto grid_z = z_vals.view({ 1, 1,-1}).expand({X, Y, Z});
    auto grid_coords = torch::stack({grid_x, grid_y, grid_z}, /*dim=*/-1).contiguous();
    grid_coords = grid_coords + 0.5f; // same offset as before

    // list the .pt/.pth files in the folder
    std::vector<fs::path> files;
    for (auto& p : fs::directory_iterator(data_folder)) {
        if (!p.is_regular_file()) continue;
        auto ext = p.path().extension().string();
        if (ext == ".pt" || ext == ".pth") files.push_back(p.path());
    }
    std::sort(files.begin(), files.end());
    std::cout << "[info] found " << files.size() << " files in " << data_folder << "\n";
    c10::Device dev0(torch::kCUDA, 0);            // target device: cuda:0
    // loop over the samples
    for (size_t i = 0; i < files.size(); ++i) {
        const std::string path = files[i].string();
        try {
            std::cout << "\n==== [" << (i+1) << "/" << files.size() << "] " << path << " ====\n";
            auto vin = load_four_tensors_from_pth(path);

            // covariances: the CUDA forward uses (N,9), the CPU reference uses (N,3,3)
            torch::Tensor cov9, cov33;
            if (vin.covariances.dim() == 2 && vin.covariances.size(1) == 9) {
                cov9  = vin.covariances.contiguous().to(dev0);
                cov33 = vin.covariances.view({vin.covariances.size(0), 3, 3});
            } else if (vin.covariances.dim() == 3 && vin.covariances.size(1) == 3 && vin.covariances.size(2) == 3) {
                cov33 = vin.covariances.contiguous().to(dev0);
                cov9  = vin.covariances.view({vin.covariances.size(0), 9});
            } else {
                throw std::runtime_error("unexpected covariances shape");
            }

            // ===== CUDA forward =====
            auto tup_fwd = cuda_gaussian_voxelizer_forward(
                vin.means3d.contiguous().to(dev0), vin.opacities.contiguous().to(dev0), cov9, voxel_size_tensor.contiguous().to(dev0), voxel_range_tensor.contiguous().to(dev0), vin.features.contiguous().to(dev0));

            auto density_tensor             = std::get<0>(tup_fwd); // [X,Y,Z]
            auto cov_inv_tensor             = std::get<1>(tup_fwd);
            auto valid_voxel_lb             = std::get<2>(tup_fwd);
            auto valid_voxel_ub             = std::get<3>(tup_fwd);
            auto num_voxels_per_gaussian    = std::get<4>(tup_fwd);
            auto grid_feats                 = std::get<5>(tup_fwd);
            auto g2t_offsets                = std::get<6>(tup_fwd);
            auto g2t_tile_ids               = std::get<7>(tup_fwd);
            auto forward_time               = std::get<8>(tup_fwd);

            auto out_reshaped = density_tensor.view({dims[0], dims[1], dims[2], 1});
            std::cout << "CUDA output shape: " << out_reshaped.sizes() << "\n";

            // ===== CUDA backward =====
            const int Cf = static_cast<int>(vin.features.size(1));
            auto grad_density_output = torch::ones({dims[0], dims[1], dims[2]}, density_tensor.options());
            auto grad_feats_output   = torch::zeros({dims[0], dims[1], dims[2], Cf}, density_tensor.options());

            auto tup_bwd = cuda_gaussian_voxelizer_backward(
                /*grad_density_output*/ grad_density_output,
                /*grad_feats_output  */ grad_feats_output,
                /*means3D            */ vin.means3d,
                /*opacity            */ vin.opacities,
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
                /*features(opt)      */ vin.features
            );
            auto backward_time = std::get<4>(tup_bwd);
            std::cout << "Backward done. Time (s): " << backward_time << "\n";

            // ===== CPU reference implementation =====
            auto grid_density = splat_into_3d_cpp(grid_coords, vin.means3d, vin.opacities, cov33,
                                                  voxel_range_vec, voxel_size_vec);
            std::cout << "CPU output shape: " << grid_density.sizes() << "\n";

            // ===== comparison =====
            if (out_reshaped.sizes() != grid_density.sizes()) {
                std::cerr << "Mismatch in output shapes! CUDA=" << out_reshaped.sizes()
                          << "  CPU=" << grid_density.sizes() << "\n";
            } else {
                auto diff      = (out_reshaped - grid_density).abs();
                float max_diff = diff.max().item().toFloat();
                float mean_diff= diff.mean().item().toFloat();
                float l2_error = torch::sqrt((diff * diff).sum()).item().toFloat();

                std::cout << "\n===== Comparison =====\n";
                std::cout << "Max absolute difference: " << max_diff  << "\n";
                std::cout << "Mean absolute difference: " << mean_diff<< "\n";
                std::cout << "L2 error: " << l2_error << "\n";

                const float tolerance = 1e-4f;
                if (max_diff < tolerance) {
                    std::cout << "✅ Outputs are numerically consistent within tolerance.\n";
                } else {
                    std::cout << "❌ Outputs differ significantly.\n";
                }
            }

            torch::cuda::synchronize();
            c10::cuda::CUDACachingAllocator::emptyCache();
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] " << e.what() << "\n";
        }
        // if(i >= 72){
        //     break;
        // }
    }

    return 0;
}
