#include "cuda_feats_loss.cuh"

namespace cudaprocess {
namespace feats_loss {

    template <typename T>
    __device__ __forceinline__ T warpReduceSum(T v, unsigned mask = 0xFFFFFFFFu) {
        for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(mask, v, offset);
        }
        return v;
    }

    template <typename T, int BLOCK>
    __device__ __forceinline__ T blockReduceSumWarp(T v) {
        static_assert(BLOCK % warp_size == 0, "BLOCK must be multiple of 32");
        const unsigned full_mask = __activemask();

        v = warpReduceSum<T>(v, full_mask);

        __shared__ T smem[BLOCK / warp_size];  // one slot per warp
        const int lane = threadIdx.x & (warp_size - 1);
        const int wid  = threadIdx.x >> 5;     // /32
        if (lane == 0) smem[wid] = v;
        __syncthreads();

        T out = (threadIdx.x < (BLOCK / warp_size)) ? smem[lane] : T(0);
        if (wid == 0) {
            unsigned mask = (BLOCK / warp_size >= warp_size)
                            ? 0xFFFFFFFFu
                            : ((1u << (BLOCK / warp_size)) - 1u);
            out = warpReduceSum<T>(out, mask);
        }
        return out;
    }

    /**
     * @brief Compute the loss for the features in the voxel grid.
     * @param voxel_label: Tensor containing the semantic labels for each voxel. Shape: (X, Y, Z)
     * @param occ_gt_mask_camera: Tensor containing the ground truth occupancy mask in camera coordinates. Shape: (X, Y, Z)
     * @param d_grid_feats: Tensor containing the features for each voxel. Shape: (X, Y, Z, C)
     * @param pseudo_dino_gt: Tensor containing the pseudo DINO features for each class. Shape: (X, Y, Z, C)
     * @return The computed loss value.
     */
    template <typename scalar_t, int BLOCK>
    __global__ void VoxelCosineLossKernel(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> voxel_label,
        const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> occ_gt_mask_camera,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_grid_feats,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> pseudo_dino_gt,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_voxel_loss
        ) {
        static_assert(BLOCK > 0 && (BLOCK & (BLOCK - 1)) == 0, "BLOCK must be power-of-two");
        static_assert(BLOCK % 32 == 0, "BLOCK must be a multiple of 32 for warp reductions");

        const int X = d_grid_feats.size(0);
        const int Y = d_grid_feats.size(1);
        const int Z = d_grid_feats.size(2);
        const int C = d_grid_feats.size(3);

        const int x = blockIdx.x;
        const int y = blockIdx.y;
        const int z = blockIdx.z;
        if (x >= X || y >= Y || z >= Z) return;

        // Intersection: visible && labeled
        const bool valid = occ_gt_mask_camera[x][y][z] &&
                        (static_cast<float>(voxel_label[x][y][z]) > 0.f);

        if (!valid) {
            // Write zero once per voxel so the reducer can just sum
            if (threadIdx.x == 0) d_voxel_loss[x][y][z] = 0.f;
            return;
        }

        // Parallel accumulation over channels (threads stride over C)
        float dot_local  = 0.f;
        float p_sq_local = 0.f;
        float g_sq_local = 0.f;

        for (int c = threadIdx.x; c < C; c += BLOCK) {
            const float p = static_cast<float>(d_grid_feats[x][y][z][c]);
            const float g = static_cast<float>(pseudo_dino_gt[x][y][z][c]);
            dot_local  += p * g;
            p_sq_local += p * p;
            g_sq_local += g * g;
        }

        // Warp-based block reductions (fewer synchronizations than shared-memory tree)
        const float dot_sum  = blockReduceSumWarp<float, BLOCK>(dot_local);
        const float p_sq_sum = blockReduceSumWarp<float, BLOCK>(p_sq_local);
        const float g_sq_sum = blockReduceSumWarp<float, BLOCK>(g_sq_local);

        // Only thread 0 holds the full block sum (warp0/lane0)
        if (threadIdx.x == 0) {
            const float p_n = sqrtf(fmaxf(p_sq_sum, eps));
            const float g_n = sqrtf(fmaxf(g_sq_sum, eps));
            const float cos_sim = dot_sum / (p_n * g_n + eps);
            d_voxel_loss[x][y][z] = 1.f - cos_sim; // per-voxel cosine loss
        }
    }

    
    template <int BLOCK>
    __global__ void ReduceLossAndCountKernel(
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> loss_map,    // [X,Y,Z]
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> voxel_label, // [X,Y,Z]
        const torch::PackedTensorAccessor32<bool, 3,torch::RestrictPtrTraits> occ_mask,    // [X,Y,Z]
        float* __restrict__ loss_out,
        int*   __restrict__ count_out)
    {
        static_assert(BLOCK > 0 &&
                    (BLOCK & (BLOCK - 1)) == 0,
                    "BLOCK must be power-of-two");
        static_assert(BLOCK % warp_size == 0,
                    "BLOCK must be multiple of warp_size (32)");

        const int X = loss_map.size(0);
        const int Y = loss_map.size(1);
        const int Z = loss_map.size(2);
        const int N = X * Y * Z;

        float loss_sum = 0.f;
        int   count_sum = 0;

        for (int i = blockIdx.x * BLOCK + threadIdx.x; i < N; i += BLOCK * gridDim.x) {
            int z  = i % Z;
            int xy = i / Z;
            int y  = xy % Y;
            int x  = xy / Y;

            loss_sum += loss_map[x][y][z];
            count_sum += (occ_mask[x][y][z] && (voxel_label[x][y][z] > 0.f)) ? 1 : 0;
        }

        float loss_sum_block = blockReduceSumWarp<float, BLOCK>(loss_sum);
        int   count_sum_block = blockReduceSumWarp<int,   BLOCK>(count_sum);

        if (threadIdx.x == 0) {
            atomicAdd(loss_out,  loss_sum_block);
            atomicAdd(count_out, count_sum_block);
        }
    }


    /**
     * @brief Compute the loss for the features in the voxel grid.
     * @param voxel_label: Tensor containing the semantic labels for each voxel. Shape: (X, Y, Z)
     * @param occ_gt_mask_camera: Tensor containing the ground truth occupancy mask in camera coordinates. Shape: (X, Y, Z)
     * @param d_grid_feats: Tensor containing the features for each voxel. Shape: (X, Y, Z, C)
     * @param pseudo_dino_gt: Tensor containing the pseudo DINO features for each class. Shape: (X, Y, Z, C)
     * @return The computed loss value.
     */
    std::tuple<torch::Tensor, int, float> CudaFeatsCosinelossForward(const torch::Tensor& voxel_label, 
                          const torch::Tensor& occ_gt_mask_camera, 
                          const torch::Tensor& d_grid_feats, 
                          const torch::Tensor& pseudo_dino_gt) {
        TORCH_CHECK(voxel_label.scalar_type()==at::kFloat && voxel_label.is_contiguous() && voxel_label.is_cuda());
        TORCH_CHECK(occ_gt_mask_camera.scalar_type()==at::kBool && occ_gt_mask_camera.is_contiguous() && occ_gt_mask_camera.is_cuda()); 
        TORCH_CHECK(d_grid_feats.scalar_type()==at::kFloat && d_grid_feats.is_contiguous() && d_grid_feats.is_cuda());
        TORCH_CHECK(pseudo_dino_gt.scalar_type()==at::kFloat && pseudo_dino_gt.is_contiguous() && pseudo_dino_gt.is_cuda());
        int voxel_dim_x = voxel_label.size(0);
        int voxel_dim_y = voxel_label.size(1);
        int voxel_dim_z = voxel_label.size(2);
        int feats_dim = d_grid_feats.size(3);

        auto s = at::cuda::getCurrentCUDAStream();

        auto float_opts = d_grid_feats.options().dtype(torch::kFloat32);
        auto int32_opts = d_grid_feats.options().dtype(torch::kInt32);

        torch::Tensor d_voxel_loss = torch::zeros({voxel_dim_x, voxel_dim_y, voxel_dim_z}, float_opts);
        
        #ifdef CUDA_TIMER_ENABLED
        cudaEvent_t cuda_start, cuda_stop;
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);

        cudaEventRecord(cuda_start, s.stream());
        #endif
        

        // int threads_per_block = feats_dim;
        // int threads_per_block = feats_dim / 2;
        constexpr int threads_per_block = 256;

        dim3 grid(voxel_dim_x, voxel_dim_y, voxel_dim_z);

        VoxelCosineLossKernel<float, threads_per_block><<<grid, threads_per_block, 0, s.stream()>>>(
                voxel_label.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                occ_gt_mask_camera.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
                d_grid_feats.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                pseudo_dino_gt.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                d_voxel_loss.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
            );
        
        // float loss_sum = 0.f;
        // int count_sum = 0;
        auto loss_sum = torch::zeros({1}, float_opts);   // device tensor [1]
        auto count_sum = torch::zeros({1}, int32_opts);
        constexpr int BLOCK = 256;
        const int GRID = std::min((voxel_dim_x * voxel_dim_y * voxel_dim_z + BLOCK - 1) / BLOCK, 65535);
        ReduceLossAndCountKernel<BLOCK><<<GRID, BLOCK, 0, s.stream()>>>(
            d_voxel_loss.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            voxel_label.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            occ_gt_mask_camera.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<int>());

        float h_loss = loss_sum.item<float>();
        int   h_cnt  = count_sum.item<int>();
        float mean_loss = (h_cnt > 0) ? (h_loss / static_cast<float>(h_cnt)) : 0.0f;

        auto mean_loss_tensor = torch::scalar_tensor(
            mean_loss,
            float_opts
        );
        
        // return mean_loss;
        #ifdef CUDA_TIMER_ENABLED
        cudaEventRecord(cuda_stop, s.stream());
        cudaEventSynchronize(cuda_stop);
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, cuda_start, cuda_stop);
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
        // std::cout << "Forward GPU Time: " << gpu_time << " ms" << std::endl;
        return std::make_tuple(mean_loss_tensor, h_cnt, gpu_time / 1000.0f);
        #else
        return std::make_tuple(mean_loss_tensor, h_cnt, 0.0f);
        #endif
    }

    // Backward: compute grad wrt predicted features p
    template <typename scalar_t, int BLOCK>
    __global__ void CosineLossBackwardKernel(
        // inputs
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_grid_feats,    // [X,Y,Z,C]
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> pseudo_dino_gt,    // [X,Y,Z,C]
        const torch::PackedTensorAccessor32<float,   3,torch::RestrictPtrTraits> label,// [X,Y,Z]
        const torch::PackedTensorAccessor32<bool,    3,torch::RestrictPtrTraits> mask, // [X,Y,Z]
        // outputs
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_p,     // [X,Y,Z,C]
        // scalars
        const float inv_count,   // 1.0f / max(N_valid,1)
        const float grad_out
        )
    {
        static_assert(BLOCK % 32 == 0, "BLOCK must be multiple of 32");

        const int X = d_grid_feats.size(0), Y = d_grid_feats.size(1), Z = d_grid_feats.size(2), C = d_grid_feats.size(3);
        const int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z;
        if (x >= X || y >= Y || z >= Z) return;

        const bool valid = mask[x][y][z] && (label[x][y][z] > 0.f);
        if (!valid) {
            // zero grad for this voxel
            for (int c = threadIdx.x; c < C; c += BLOCK) {
                grad_p[x][y][z][c] = static_cast<scalar_t>(0);
            }
            return;
        }

        // 1) parallel reductions to get dot, p_sq, g_sq
        float dot_local = 0.f, p_sq_local = 0.f, g_sq_local = 0.f;
        for (int c = threadIdx.x; c < C; c += BLOCK) {
            const float pv = static_cast<float>(d_grid_feats[x][y][z][c]);
            const float gv = static_cast<float>(pseudo_dino_gt[x][y][z][c]);
            dot_local  += pv * gv;
            p_sq_local += pv * pv;
            g_sq_local += gv * gv;
        }
        const float dot  = blockReduceSumWarp<float, BLOCK>(dot_local);
        const float p_sq = blockReduceSumWarp<float, BLOCK>(p_sq_local);
        const float g_sq = blockReduceSumWarp<float, BLOCK>(g_sq_local);

        // 2) compute coefficients (only thread 0 does it)
        __shared__ float coef1, coef2; // grad = grad_out*inv_count*(coef1*g_c + coef2*p_c)
        if (threadIdx.x == 0) {
            const float a = sqrtf(fmaxf(p_sq, eps));
            const float b = sqrtf(fmaxf(g_sq, eps));
            const float D = a * b + eps;

            float c1 = -1.0f / D;                 // multiplies g
            float c2 = 0.0f;                      // multiplies p
            if (p_sq > eps) {
                c2 = (dot * b) / (a * D * D);
            }
            coef1 = c1 * (grad_out * inv_count);
            coef2 = c2 * (grad_out * inv_count);
        }
        __syncthreads();

        // 3) write grad per channel
        for (int c = threadIdx.x; c < C; c += BLOCK) {
            const float gv = static_cast<float>(pseudo_dino_gt[x][y][z][c]);
            const float pv = static_cast<float>(d_grid_feats[x][y][z][c]);
            const float gp = coef1 * gv + coef2 * pv;     // dL/dp_c
            grad_p[x][y][z][c] = static_cast<scalar_t>(gp);
        }
    }

    /**
     * @brief Calculate the gradient of the cosine loss with respect to the voxel features.
     * @param voxel_label: Tensor containing the semantic labels for each voxel. Shape: (X, Y, Z)
     * @param occ_gt_mask_camera: Tensor containing the ground truth occupancy mask in camera coordinates. Shape: (X, Y, Z)
     * @param d_grid_feats: Tensor containing the features for each voxel. Shape: (X, Y, Z, C)
     * @param pseudo_dino_gt: Tensor containing the pseudo DINO features for each class. Shape: (X, Y, Z, C)
     * @return A tuple containing the computed loss value and the time taken for the computation.
     */
    std::tuple<torch::Tensor, float> CudaFeatsCosinelossBackward(
        const torch::Tensor& grad_output,
        const torch::Tensor& voxel_label, 
        const torch::Tensor& occ_gt_mask_camera, 
        const torch::Tensor& d_grid_feats, 
        const torch::Tensor& pseudo_dino_gt,
        const int num_valid_voxel) 
    {
        float inv_count = (num_valid_voxel > 0) ? (1.0f / static_cast<float>(num_valid_voxel)) : 0.0f;

        // the gradient of predicted features (d_grid_feats)
        auto grad_p = torch::zeros_like(d_grid_feats);

        int voxel_dim_x = voxel_label.size(0);
        int voxel_dim_y = voxel_label.size(1);
        int voxel_dim_z = voxel_label.size(2);
        int feats_dim = d_grid_feats.size(3);

        auto s = at::cuda::getCurrentCUDAStream();

        #ifdef CUDA_TIMER_ENABLED
        cudaEvent_t cuda_start, cuda_stop;
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);

        cudaEventRecord(cuda_start, s.stream());
        #endif

        constexpr int KBLOCK = 256;
        dim3 kGrid(voxel_dim_x, voxel_dim_y, voxel_dim_z);
        dim3 kBlock(KBLOCK);

        CosineLossBackwardKernel<float, KBLOCK><<<kGrid, kBlock>>>(
            d_grid_feats.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            pseudo_dino_gt.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            voxel_label.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            occ_gt_mask_camera.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
            grad_p.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            inv_count,
            grad_output.item<float>()
        );

        // return std::make_tuple(grad_p, 0.0f);

        #ifdef CUDA_TIMER_ENABLED
        cudaEventRecord(cuda_stop, s.stream());
        cudaEventSynchronize(cuda_stop);
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, cuda_start, cuda_stop);
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
        // std::cout << "Backwward GPU Time: " << gpu_time << " ms" << std::endl;
        return std::make_tuple(grad_p, gpu_time / 1000.0f);
        #else
        return std::make_tuple(grad_p, 0.0f);
        #endif
    }
}  // namespace feats_loss
}