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

// LIBTORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cudaprocess {
    namespace feats_loss {
        std::tuple<torch::Tensor, int, float> CudaFeatsCosinelossForward(const torch::Tensor& voxel_label, 
                                                const torch::Tensor& occ_gt_mask_camera, 
                                                const torch::Tensor& d_grid_feats, 
                                                const torch::Tensor& pseudo_dino_gt);

        std::tuple<torch::Tensor, float> CudaFeatsCosinelossBackward(
                                                            const torch::Tensor& grad_output,
                                                            const torch::Tensor& voxel_label, 
                                                            const torch::Tensor& occ_gt_mask_camera, 
                                                            const torch::Tensor& d_grid_feats, 
                                                            const torch::Tensor& pseudo_dino_gt,
                                                            const int num_valid_voxel);
    }  // namespace feats_loss
}