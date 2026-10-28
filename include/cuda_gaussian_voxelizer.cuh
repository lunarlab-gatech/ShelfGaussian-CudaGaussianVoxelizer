#pragma once
#include <cstdio>
#include <tuple>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_utils.cuh"
#include "utils.cuh"
#include "type.cuh"
#include <cuda_runtime.h>

// #include <nvToolsExt.h>

// #include <cooperative_groups.h>

// LIBTORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// using namespace cooperative_groups;

namespace cudaprocess{
namespace gaussian_voxelizer{
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float> cuda_gaussian_voxelizer_forward(const torch::Tensor& means3D,
                                    // const torch::Tensor& semantics, 
                                    const torch::Tensor& opacity,
                                    // const torch::Tensor& scale, 
                                    const torch::Tensor& covariances,
                                    const torch::Tensor& voxel_size,
                                    const torch::Tensor& voxel_range,
                                    const c10::optional<torch::Tensor> features = c10::nullopt);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float> cuda_gaussian_voxelizer_backward(const torch::Tensor& grad_density_output,
                                                                                            const torch::Tensor& grad_feats_output,
                                                                                            const torch::Tensor& means3D, 
                                                                                            const torch::Tensor& opacity,
                                                                                            const torch::Tensor& cov_inv,
                                                                                            const torch::Tensor& voxel_size,
                                                                                            const torch::Tensor& voxel_range,
                                                                                            const torch::Tensor& valid_voxel_lb,
                                                                                            const torch::Tensor& valid_voxel_ub,
                                                                                            const torch::Tensor& num_voxels_per_gaussian,
                                                                                            const torch::Tensor& g2t_offsets,
                                                                                            const torch::Tensor& g2t_tile_ids,
                                                                                            const torch::Tensor& density,
                                                                                            const torch::Tensor& grid_feats,
                                                                                            const c10::optional<torch::Tensor> features = c10::nullopt
                                                                                            );
}  // namespace gaussian_voxelizer
} // namespace cudaprocess


