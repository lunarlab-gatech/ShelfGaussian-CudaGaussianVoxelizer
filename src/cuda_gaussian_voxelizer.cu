#include "cuda_gaussian_voxelizer.cuh"

// #include "utils.cuh"
#include <cub/cub.cuh>
namespace cudaprocess {
namespace gaussian_voxelizer {
    template <typename scalar_t>
    __global__ void ComputeGaussian2VoxelTile(
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mean3d,
                                            // const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussian_covariances,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_gaussian_opacity,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_range,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_size,
                                            torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_grid_density_tensor,
                                            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussians_cov_inv,
                                            torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> d_num_voxels_per_gaussian,
                                            const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_lower_boundary,
                                            const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_upper_boundary,
                                            const int voxel_dim_x,
                                            const int voxel_dim_y,
                                            const int voxel_dim_z,
                                            const int max_tiles_x,
                                            const int max_tiles_y,
                                            const int max_tiles_z,
                                            const int* __restrict__ active_tiles,
                                            const int* __restrict__ runs_offsets,
                                            const int* __restrict__ d_sorted_gaussian_ids,
                                            const int num_active_tiles,
                                            const int num_runs,
                                            const int num_gaussians
                                            // const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_features,
                                            // torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_grid_feats
                                            ) {
        // decode the tile to global tile coordinates
        int tile_idx = blockIdx.x;
        if (tile_idx >= num_active_tiles) return;
        const int global_tile_id = active_tiles[tile_idx];
        int tx = global_tile_id / (max_tiles_y * max_tiles_z);
        int rem = global_tile_id % (max_tiles_y * max_tiles_z);
        int ty = rem / max_tiles_z;
        int tz = rem % max_tiles_z;

        // convert threadIdx.x to local voxel index in the tile
        int local_voxel_idx = threadIdx.x;
        if (local_voxel_idx >= TILE_X * TILE_Y * TILE_Z) return;

        int local_voxel_x_idx = local_voxel_idx / (TILE_Y * TILE_Z);
        int rem1 = local_voxel_idx % (TILE_Y * TILE_Z);
        int local_voxel_y_idx = rem1 / TILE_Z;
        int local_voxel_z_idx = rem1 % TILE_Z;

        // convert local voxel index to global voxel index
        const int x = tx * TILE_X + local_voxel_x_idx;
        const int y = ty * TILE_Y + local_voxel_y_idx;
        const int z = tz * TILE_Z + local_voxel_z_idx;

        if (x >= voxel_dim_x || y >= voxel_dim_y || z >= voxel_dim_z){
            #ifdef DEBUG_MODE
            printf("Error: voxel (%d,%d,%d) is out of bound (%d,%d,%d)\n", x, y, z, voxel_dim_x, voxel_dim_y, voxel_dim_z);
            #endif
            return;
        }

        // calculate the center of the voxel
        float3 voxel_center = make_float3(
            d_voxel_range[0] + (x + 0.5f) * d_voxel_size[0],
            d_voxel_range[1] + (y + 0.5f) * d_voxel_size[1],
            d_voxel_range[2] + (z + 0.5f) * d_voxel_size[2]);

        // get the gaussians that are affecting the current tile
        int start_run = runs_offsets[tile_idx];
        int end_run = runs_offsets[tile_idx + 1];
        // int start_run = runs_offsets[global_tile_id];
        // int end_run   = runs_offsets[global_tile_id + 1];

        float dens_acc = 0.f;

        // iterate over all the gaussians that are affecting the current tile
        for (int run_idx = start_run; run_idx < end_run; ++run_idx) {
            // if (run_idx >= num_runs) break;
            // if(run_idx < 0 || run_idx >= num_runs) continue;
            int gaussian_idx = d_sorted_gaussian_ids[run_idx];
            if(gaussian_idx < 0 || gaussian_idx >= num_gaussians) continue;
            if (d_num_voxels_per_gaussian[gaussian_idx] == 0) continue;

            const int lx = d_valid_voxel_lower_boundary[gaussian_idx][0];
            const int ly = d_valid_voxel_lower_boundary[gaussian_idx][1];
            const int lz = d_valid_voxel_lower_boundary[gaussian_idx][2];
            const int ux = d_valid_voxel_upper_boundary[gaussian_idx][0];
            const int uy = d_valid_voxel_upper_boundary[gaussian_idx][1];
            const int uz = d_valid_voxel_upper_boundary[gaussian_idx][2];

            // check if the voxel is within the valid boundary of the gaussian
            if (x < lx || x > ux || y < ly || y > uy || z < lz || z > uz) {
                continue;
            }

            float3 mu = make_float3(
                mean3d[gaussian_idx][0],
                mean3d[gaussian_idx][1],
                mean3d[gaussian_idx][2]
            );
            const float opacity = d_gaussian_opacity[gaussian_idx];

            // const float* __restrict__ cov_inv = d_gaussians_cov_inv[gaussian_idx].data();
            const scalar_t* __restrict__ cov_inv = &d_gaussians_cov_inv[gaussian_idx][0];

            float dx[3] = {voxel_center.x - mu.x,
                        voxel_center.y - mu.y,
                        voxel_center.z - mu.z};
            
            // float maha = 0.0f;
            // #pragma unroll
            // for (int i = 0; i < 3; ++i)
            //     #pragma unroll
            //     for (int j = 0; j < 3; ++j)
            //         // maha += dx[i] * cov_inv[i * 3 + j] * dx[j];
            //         maha = fmaf(dx[i] * cov_inv[i * 3 + j], dx[j], maha);
            float qx = cov_inv[0] * dx[0] + cov_inv[1] * dx[1] + cov_inv[2] * dx[2];
            float qy = cov_inv[3] * dx[0] + cov_inv[4] * dx[1] + cov_inv[5] * dx[2];
            float qz = cov_inv[6] * dx[0] + cov_inv[7] * dx[1] + cov_inv[8] * dx[2];
            float maha = dx[0] * qx + dx[1] * qy + dx[2] * qz;
            // float qy = cov_inv[3]*dx0 + cov_inv[4]*dx1 + cov_inv[5]*dx2;
            // float qz = cov_inv[6]*dx0 + cov_inv[7]*dx1 + cov_inv[8]*dx2;
            // float maha = dx0*qx + dx1*qy + dx2*qz;

            
            float exp_term = __expf(-0.5f * maha);  // Use __expf for better performance on CUDA
            // float exp_term = expf(-0.5f * maha);  // Use __expf for better performance on CUDA
            float density = opacity * exp_term;

            dens_acc += density;
        }

        d_grid_density_tensor[x][y][z] += dens_acc;
    }
    // // float atomicMin
    // // just only support float > 0.0
    // __device__ __forceinline__ float atomicMin_float(float *address, float val)
    // {
    //     int ret = __float_as_int(*address);
    //     while(val < __int_as_float(ret))
    //     {
    //         int old = ret;
    //         if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
    //             break;
    //     }
    //     return __int_as_float(ret);
    // }

    // // float atomicMax
    // // just only support float > 0.0
    // __device__ __forceinline__ float atomicMax_float(float *address, float val)
    // {
    //     int ret = __float_as_int(*address);
    //     while(val > __int_as_float(ret))
    //     {
    //         int old = ret;
    //         if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
    //             break;
    //     }
    //     return __int_as_float(ret);
    // }

    // __device__ __forceinline__ void invert_3x3_flat(const float A[9], float Ainv[9], float eps = 1e-6f) {
    //     float det = A[0]*(A[4]*A[8] - A[5]*A[7])
    //             - A[1]*(A[3]*A[8] - A[5]*A[6])
    //             + A[2]*(A[3]*A[7] - A[4]*A[6]);

    //     if (fabsf(det) < eps) det = eps;
    //     float inv_det = 1.0f / det;

    //     Ainv[0] =  (A[4]*A[8] - A[5]*A[7]) * inv_det;
    //     Ainv[1] = -(A[1]*A[8] - A[2]*A[7]) * inv_det;
    //     Ainv[2] =  (A[1]*A[5] - A[2]*A[4]) * inv_det;

    //     Ainv[3] = -(A[3]*A[8] - A[5]*A[6]) * inv_det;
    //     Ainv[4] =  (A[0]*A[8] - A[2]*A[6]) * inv_det;
    //     Ainv[5] = -(A[0]*A[5] - A[2]*A[3]) * inv_det;

    //     Ainv[6] =  (A[3]*A[7] - A[4]*A[6]) * inv_det;
    //     Ainv[7] = -(A[0]*A[7] - A[1]*A[6]) * inv_det;
    //     Ainv[8] =  (A[0]*A[4] - A[1]*A[3]) * inv_det;
    // }

    __global__ void CheckTileIds(const int* ids, int M, int num_tiles,
                             int* bad, int* first_bad_idx, int* first_bad_val){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < M) {
            int t = ids[i];
            if (t < 0 || t >= num_tiles) {
                if (atomicCAS(first_bad_idx, -1, i) == -1) {
                    *first_bad_val = t;
                }
                atomicAdd(bad, 1);
            }
        }
    }


    __device__ __forceinline__ void invert_3x3_flat(const float A[9], float Ainv[9], float eps = 1e-6f) {
        float det = A[0]*(A[4]*A[8] - A[5]*A[7])
                - A[1]*(A[3]*A[8] - A[5]*A[6])
                + A[2]*(A[3]*A[7] - A[4]*A[6]);

        if (fabsf(det) < eps) det = eps;
        float inv_det = 1.0f / det;

        Ainv[0] =  (A[4]*A[8] - A[5]*A[7]) * inv_det;
        Ainv[1] = -(A[1]*A[8] - A[2]*A[7]) * inv_det;
        Ainv[2] =  (A[1]*A[5] - A[2]*A[4]) * inv_det;

        Ainv[3] = -(A[3]*A[8] - A[5]*A[6]) * inv_det;
        Ainv[4] =  (A[0]*A[8] - A[2]*A[6]) * inv_det;
        Ainv[5] = -(A[0]*A[5] - A[2]*A[3]) * inv_det;

        Ainv[6] =  (A[3]*A[7] - A[4]*A[6]) * inv_det;
        Ainv[7] = -(A[0]*A[7] - A[1]*A[6]) * inv_det;
        Ainv[8] =  (A[0]*A[4] - A[1]*A[3]) * inv_det;
    }


    __global__ void ScatterCounts(const int* d_unique_tiles, const int* cnts, int unique_tiles_count, int* full){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < unique_tiles_count) full[ d_unique_tiles[i] ] = cnts[i];
    }

    __global__ void FillG2TKernel(
        const int* __restrict__ tile_lo,      // [G,3]
        const int* __restrict__ tile_hi,      // [G,3]
        // const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_tile_lb,
        // const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_tile_hb,
        const int* __restrict__ g2t_offsets,  // [G+1]
        int tiles_x, int tiles_y, int tiles_z, int num_gaussians,
        int* __restrict__ g2t_tile_ids,        // [M]
        int* __restrict__ g2t_gaussian_ids     // [M]
    ){
        int gaussian_id = blockIdx.x;
        if (gaussian_id >= num_gaussians) return;
        // int tx0 = d_tile_lb[gaussian_id][0], ty0 = d_tile_lb[gaussian_id][1], tz0 = d_tile_lb[gaussian_id][2];
        // int tx1 = d_tile_hb[gaussian_id][0], ty1 = d_tile_hb[gaussian_id][1], tz1 = d_tile_hb[gaussian_id][2];
        int tx0 = tile_lo[gaussian_id * dim3d + 0], ty0 = tile_lo[gaussian_id * dim3d + 1], tz0 = tile_lo[gaussian_id * dim3d + 2];
        int tx1 = tile_hi[gaussian_id * dim3d + 0], ty1 = tile_hi[gaussian_id * dim3d + 1], tz1 = tile_hi[gaussian_id * dim3d + 2];

        // tx0 = max(0, min(tx0, tiles_x-1));
        // ty0 = max(0, min(ty0, tiles_y-1));
        // tz0 = max(0, min(tz0, tiles_z-1));
        // tx1 = max(0, min(tx1, tiles_x-1));
        // ty1 = max(0, min(ty1, tiles_y-1));
        // tz1 = max(0, min(tz1, tiles_z-1));
        // if (tx1 < tx0 || ty1 < ty0 || tz1 < tz0) return;


        int dx = tx1 - tx0 + 1, dy = ty1 - ty0 + 1, dz = tz1 - tz0 + 1;
        int count = dx * dy * dz;
        int base  = g2t_offsets[gaussian_id];

        if (dx <= 0 || dy <= 0 || dz <= 0){
            #ifdef DEBUG_MODE
            printf("Error Gaussian %d has invalid tile boundary: [%d,%d,%d] to [%d,%d,%d]\n", gaussian_id, tx0, ty0, tz0, tx1, ty1, tz1);
            #endif
            return;
        }

        // int chunk_size = (blockDim.x + count - 1) / blockDim.x;

        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            int lx = i / (dy*dz);
            int r  = i % (dy*dz);
            int ly = r / dz;
            int lz = r % dz;

            int tx = tx0 + lx, ty = ty0 + ly, tz = tz0 + lz;
            if ((unsigned)tx >= (unsigned)tiles_x ||
                (unsigned)ty >= (unsigned)tiles_y ||
                (unsigned)tz >= (unsigned)tiles_z) {
                    #ifdef DEBUG_MODE
                    printf("Error Gaussian %d has out-of-bound tile index: [%d,%d,%d] from [%d,%d,%d] to [%d,%d,%d]\n", gaussian_id, tx, ty, tz, tx0, ty0, tz0, tx1, ty1, tz1);
                    #endif
                    continue;
                }
            int tid = (tx * tiles_y + ty) * tiles_z + tz;   // 3D->1D
            g2t_tile_ids[base + i] = tid;
            g2t_gaussian_ids[base + i] = gaussian_id;
        }
    }

    // __global__ void FillG2TKernel(
    //     const int* __restrict__ tile_lo,      // [G,3]
    //     const int* __restrict__ tile_hi,      // [G,3]
    //     const int* __restrict__ g2t_offsets,  // [G+1] (exclusive-scan offsets)
    //     int tiles_x, int tiles_y, int tiles_z, int num_gaussians,
    //     int* __restrict__ g2t_tile_ids,       // [M]
    //     int* __restrict__ g2t_gaussian_ids,   // [M]
    //     int* __restrict__ written_counts      // [G]  (new: records how many entries each G actually wrote)
    // ){
    //     int g = blockIdx.x;
    //     if (g >= num_gaussians) return;

    //     // get the tile boundary (access [G,3] in packed row-major order)
    //     int tx0 = tile_lo[g*3 + 0];
    //     int ty0 = tile_lo[g*3 + 1];
    //     int tz0 = tile_lo[g*3 + 2];
    //     int tx1 = tile_hi[g*3 + 0];
    //     int ty1 = tile_hi[g*3 + 1];
    //     int tz1 = tile_hi[g*3 + 2];

    //     // clamp again (for safety)
    //     tx0 = max(0, min(tx0, tiles_x-1));  ty0 = max(0, min(ty0, tiles_y-1));  tz0 = max(0, min(tz0, tiles_z-1));
    //     tx1 = max(0, min(tx1, tiles_x-1));  ty1 = max(0, min(ty1, tiles_y-1));  tz1 = max(0, min(tz1, tiles_z-1));

    //     if (tx1 < tx0 || ty1 < ty0 || tz1 < tz0) { if (threadIdx.x == 0) written_counts[g] = 0; return; }

    //     const int base = g2t_offsets[g];

    //     __shared__ int write_ptr;
    //     if (threadIdx.x == 0) write_ptr = 0;
    //     __syncthreads();

    //     const int dx = tx1 - tx0 + 1, dy = ty1 - ty0 + 1, dz = tz1 - tz0 + 1;
    //     const int total = dx * dy * dz;

    //     for (int i = threadIdx.x; i < total; i += blockDim.x) {
    //         int lx = i / (dy*dz);
    //         int r  = i % (dy*dz);
    //         int ly = r / dz;
    //         int lz = r % dz;

    //         int tx = tx0 + lx, ty = ty0 + ly, tz = tz0 + lz;
    //         // normally this check is always true, kept for safety
    //         if ((unsigned)tx >= (unsigned)tiles_x ||
    //             (unsigned)ty >= (unsigned)tiles_y ||
    //             (unsigned)tz >= (unsigned)tiles_z) continue;

    //         int tid = (tx * tiles_y + ty) * tiles_z + tz; // 3D->1D
    //         int pos = atomicAdd(&write_ptr, 1);
    //         g2t_tile_ids    [base + pos] = tid;
    //         g2t_gaussian_ids[base + pos] = g;
    //     }

    //     if (threadIdx.x == 0) {
    //         written_counts[g] = write_ptr;   // record the actual number of written entries
    //     }
    // }



    template <typename scalar_t>
    __global__ void ComputeGaussianVoxelPairTile(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> means3D,
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_covariances,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_range,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_size,
                                            const int voxel_dim_x,
                                            const int voxel_dim_y,
                                            const int voxel_dim_z,
                                            torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> d_num_voxels_per_gaussian,
                                            torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_lower_boundary,
                                            torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_upper_boundary,
                                            torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_tile_lower_boundary,
                                            torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_tile_higher_boundary,
                                            torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> d_num_tiles_per_gaussian,
                                            torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_gaussians_cov_inv,
                                            const int max_tiles_x,
                                            const int max_tiles_y,
                                            const int max_tiles_z
                                            ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int num_gaussians = means3D.size(0);
        if (idx >= num_gaussians) return;
        
        float3 mu = make_float3(means3D[idx][0], means3D[idx][1], means3D[idx][2]);
        float sigma_x = sqrtf(d_covariances[idx][0]);
        float sigma_y = sqrtf(d_covariances[idx][4]);
        float sigma_z = sqrtf(d_covariances[idx][8]);

        float mu_x = means3D[idx][0];
        float cov_xx = d_covariances[idx][0];
        float cov_yy = d_covariances[idx][4];
        float cov_zz = d_covariances[idx][8];

        // low and upper bounds for the Gaussian
        float3 lower = make_float3(mu.x - 3 * sigma_x, mu.y - 3 * sigma_y, mu.z - 3 * sigma_z);
        float3 upper = make_float3(mu.x + 3 * sigma_x, mu.y + 3 * sigma_y, mu.z + 3 * sigma_z);

        // bool fully_inside =
        //     (lower.x >= d_voxel_range[0]) &&
        //     (lower.y >= d_voxel_range[1]) &&
        //     (lower.z >= d_voxel_range[2]) &&
        //     (upper.x <= d_voxel_range[3]) &&
        //     (upper.y <= d_voxel_range[4]) &&
        //     (upper.z <= d_voxel_range[5]);

        // if (!fully_inside) {
        //     // clear/mark exactly the same way as the empty branch
        //     d_valid_voxel_lower_boundary[idx][0] = 0;
        //     d_valid_voxel_lower_boundary[idx][1] = 0;
        //     d_valid_voxel_lower_boundary[idx][2] = 0;
        //     d_valid_voxel_upper_boundary[idx][0] = -1;
        //     d_valid_voxel_upper_boundary[idx][1] = -1;
        //     d_valid_voxel_upper_boundary[idx][2] = -1;

        //     d_num_voxels_per_gaussian[idx] = 0;

        //     d_tile_lower_boundary[idx][0] = 0;
        //     d_tile_lower_boundary[idx][1] = 0;
        //     d_tile_lower_boundary[idx][2] = 0;
        //     d_tile_higher_boundary[idx][0] = -1;
        //     d_tile_higher_boundary[idx][1] = -1;
        //     d_tile_higher_boundary[idx][2] = -1;
        //     d_num_tiles_per_gaussian[idx] = 0;

        //     #pragma unroll
        //     for (int k = 0; k < 9; ++k) d_gaussians_cov_inv[idx][k] = 0.f;
        //     return;
        // }
        
        // Clamp to voxel range
        lower.x = fmaxf(lower.x, d_voxel_range[0]);
        lower.y = fmaxf(lower.y, d_voxel_range[1]);
        lower.z = fmaxf(lower.z, d_voxel_range[2]);

        upper.x = fminf(upper.x, d_voxel_range[3]);
        upper.y = fminf(upper.y, d_voxel_range[4]);
        upper.z = fminf(upper.z, d_voxel_range[5]);

        // Convert to voxel indices
        // int3 lo_idx = make_int3(
        //     (int)((lower.x - d_voxel_range[0]) / d_voxel_size[0]),
        //     (int)((lower.y - d_voxel_range[1]) / d_voxel_size[1]),
        //     (int)((lower.z - d_voxel_range[2]) / d_voxel_size[2])
        // );

        // int3 hi_idx = make_int3(
        //     (int)((upper.x - d_voxel_range[0]) / d_voxel_size[0]),
        //     (int)((upper.y - d_voxel_range[1]) / d_voxel_size[1]),
        //     (int)((upper.z - d_voxel_range[2]) / d_voxel_size[2])
        // );

        auto to_idx = [&](float v, float v0, float h){
            return (int)floorf((v - v0) / h);
        };

        int3 lo_idx = make_int3(
            to_idx(lower.x, d_voxel_range[0], d_voxel_size[0]),
            to_idx(lower.y, d_voxel_range[1], d_voxel_size[1]),
            to_idx(lower.z, d_voxel_range[2], d_voxel_size[2])
        );
        int3 hi_idx = make_int3(
            to_idx(upper.x, d_voxel_range[0], d_voxel_size[0]),
            to_idx(upper.y, d_voxel_range[1], d_voxel_size[1]),
            to_idx(upper.z, d_voxel_range[2], d_voxel_size[2])
        );

        // Clamp to grid size
        // lo_idx.x = max(lo_idx.x, 0);
        // lo_idx.y = max(lo_idx.y, 0);
        // lo_idx.z = max(lo_idx.z, 0);

        // hi_idx.x = min(hi_idx.x, voxel_dim_x - 1);
        // hi_idx.y = min(hi_idx.y, voxel_dim_y - 1);
        // hi_idx.z = min(hi_idx.z, voxel_dim_z - 1);

        lo_idx.x = max(lo_idx.x, 0);
        lo_idx.y = max(lo_idx.y, 0);
        lo_idx.z = max(lo_idx.z, 0);
        hi_idx.x = min(hi_idx.x, voxel_dim_x - 1);
        hi_idx.y = min(hi_idx.y, voxel_dim_y - 1);
        hi_idx.z = min(hi_idx.z, voxel_dim_z - 1);
        // int count = (clamped_hi_x - clamped_lo_x + 1) * (clamped_hi_y - clamped_lo_y + 1) * (clamped_hi_z - clamped_lo_z + 1);
        // // int count = (hi_idx.x - lo_idx.x + 1) * (hi_idx.y - lo_idx.y + 1) * (hi_idx.z - lo_idx.z + 1);

        bool empty = (lo_idx.x > hi_idx.x) ||
             (lo_idx.y > hi_idx.y) ||
             (lo_idx.z > hi_idx.z);

        if (empty) {
            d_valid_voxel_lower_boundary[idx][0] = 0;
            d_valid_voxel_lower_boundary[idx][1] = 0;
            d_valid_voxel_lower_boundary[idx][2] = 0;
            d_valid_voxel_upper_boundary[idx][0] = -1;
            d_valid_voxel_upper_boundary[idx][1] = -1;
            d_valid_voxel_upper_boundary[idx][2] = -1;

            d_num_voxels_per_gaussian[idx] = 0;

            d_tile_lower_boundary[idx][0] = 0;
            d_tile_lower_boundary[idx][1] = 0;
            d_tile_lower_boundary[idx][2] = 0;
            d_tile_higher_boundary[idx][0] = -1;
            d_tile_higher_boundary[idx][1] = -1;
            d_tile_higher_boundary[idx][2] = -1;
            d_num_tiles_per_gaussian[idx] = 0;

            #pragma unroll
            for (int k = 0; k < 9; ++k) d_gaussians_cov_inv[idx][k] = 0.f;
            return;
        }

        
        d_valid_voxel_lower_boundary[idx][0] = lo_idx.x;
        d_valid_voxel_lower_boundary[idx][1] = lo_idx.y;
        d_valid_voxel_lower_boundary[idx][2] = lo_idx.z;
        d_valid_voxel_upper_boundary[idx][0] = hi_idx.x;
        d_valid_voxel_upper_boundary[idx][1] = hi_idx.y;
        d_valid_voxel_upper_boundary[idx][2] = hi_idx.z;

        int count = (hi_idx.x - lo_idx.x + 1) * (hi_idx.y - lo_idx.y + 1) * (hi_idx.z - lo_idx.z + 1);
        d_num_voxels_per_gaussian[idx] = count;

        // Tile boundaries
        int3 tile_lo = make_int3(lo_idx.x / TILE_X,
                         lo_idx.y / TILE_Y,
                         lo_idx.z / TILE_Z);
        int3 tile_hi = make_int3(hi_idx.x / TILE_X,
                                hi_idx.y / TILE_Y,
                                hi_idx.z / TILE_Z);

        // clamp to the tile grid range
        tile_lo.x = max(0, tile_lo.x);
        tile_lo.y = max(0, tile_lo.y);
        tile_lo.z = max(0, tile_lo.z);
        tile_hi.x = min(max_tiles_x - 1, tile_hi.x);
        tile_hi.y = min(max_tiles_y - 1, tile_hi.y);
        tile_hi.z = min(max_tiles_z - 1, tile_hi.z);

        d_tile_lower_boundary[idx][0] = tile_lo.x;
        d_tile_lower_boundary[idx][1] = tile_lo.y;
        d_tile_lower_boundary[idx][2] = tile_lo.z;
        d_tile_higher_boundary[idx][0] = tile_hi.x;
        d_tile_higher_boundary[idx][1] = tile_hi.y;
        d_tile_higher_boundary[idx][2] = tile_hi.z;
        d_num_tiles_per_gaussian[idx] = (tile_hi.x - tile_lo.x + 1) * (tile_hi.y - tile_lo.y + 1) * (tile_hi.z - tile_lo.z + 1);

        // calculate the inverse of
        invert_3x3_flat(d_covariances[idx].data(), d_gaussians_cov_inv[idx].data());
    }

    template <typename scalar_t>
    __global__ void ComputeGaussianVoxelPair(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> means3D,
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_covariances,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_range,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_size,
                                            const int voxel_dim_x,
                                            const int voxel_dim_y,
                                            const int voxel_dim_z,
                                            torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> d_num_voxels_per_gaussian,
                                            torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_lower_boundary,
                                            torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_upper_boundary) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int num_gaussians = means3D.size(0);
        if (idx >= num_gaussians) return;
        
        float3 mu = make_float3(means3D[idx][0], means3D[idx][1], means3D[idx][2]);
        float sigma_x = sqrtf(d_covariances[idx][0]);
        float sigma_y = sqrtf(d_covariances[idx][4]);
        float sigma_z = sqrtf(d_covariances[idx][8]);

        float mu_x = means3D[idx][0];
        float cov_xx = d_covariances[idx][0];
        float cov_yy = d_covariances[idx][4];
        float cov_zz = d_covariances[idx][8];

        // low and upper bounds for the Gaussian
        float3 lower = make_float3(mu.x - 3 * sigma_x, mu.y - 3 * sigma_y, mu.z - 3 * sigma_z);
        float3 upper = make_float3(mu.x + 3 * sigma_x, mu.y + 3 * sigma_y, mu.z + 3 * sigma_z);
        
        // Clamp to voxel range
        lower.x = fmaxf(lower.x, d_voxel_range[0]);
        lower.y = fmaxf(lower.y, d_voxel_range[1]);
        lower.z = fmaxf(lower.z, d_voxel_range[2]);

        upper.x = fminf(upper.x, d_voxel_range[3]);
        upper.y = fminf(upper.y, d_voxel_range[4]);
        upper.z = fminf(upper.z, d_voxel_range[5]);

        // Convert to voxel indices
        int3 lo_idx = make_int3(
            (int)((lower.x - d_voxel_range[0]) / d_voxel_size[0]),
            (int)((lower.y - d_voxel_range[1]) / d_voxel_size[1]),
            (int)((lower.z - d_voxel_range[2]) / d_voxel_size[2])
        );

        int3 hi_idx = make_int3(
            (int)((upper.x - d_voxel_range[0]) / d_voxel_size[0]),
            (int)((upper.y - d_voxel_range[1]) / d_voxel_size[1]),
            (int)((upper.z - d_voxel_range[2]) / d_voxel_size[2])
        );

        // Clamp to grid size
        // lo_idx.x = max(lo_idx.x, 0);
        // lo_idx.y = max(lo_idx.y, 0);
        // lo_idx.z = max(lo_idx.z, 0);

        // hi_idx.x = min(hi_idx.x, voxel_dim_x - 1);
        // hi_idx.y = min(hi_idx.y, voxel_dim_y - 1);
        // hi_idx.z = min(hi_idx.z, voxel_dim_z - 1);

        lo_idx.x = max(lo_idx.x, 0);
        lo_idx.y = max(lo_idx.y, 0);
        lo_idx.z = max(lo_idx.z, 0);
        hi_idx.x = min(hi_idx.x, voxel_dim_x - 1);
        hi_idx.y = min(hi_idx.y, voxel_dim_y - 1);
        hi_idx.z = min(hi_idx.z, voxel_dim_z - 1);
        // int count = (clamped_hi_x - clamped_lo_x + 1) * (clamped_hi_y - clamped_lo_y + 1) * (clamped_hi_z - clamped_lo_z + 1);
        // // int count = (hi_idx.x - lo_idx.x + 1) * (hi_idx.y - lo_idx.y + 1) * (hi_idx.z - lo_idx.z + 1);

        
        d_valid_voxel_lower_boundary[idx][0] = lo_idx.x;
        d_valid_voxel_lower_boundary[idx][1] = lo_idx.y;
        d_valid_voxel_lower_boundary[idx][2] = lo_idx.z;
        d_valid_voxel_upper_boundary[idx][0] = hi_idx.x;
        d_valid_voxel_upper_boundary[idx][1] = hi_idx.y;
        d_valid_voxel_upper_boundary[idx][2] = hi_idx.z;

        int count = (hi_idx.x - lo_idx.x + 1) * (hi_idx.y - lo_idx.y + 1) * (hi_idx.z - lo_idx.z + 1);
        d_num_voxels_per_gaussian[idx] = count;
    }

    template <typename scalar_t>
    __global__ void ComputeGaussian2Voxel(
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mean3d,
                                            // const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussian_covariances,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_gaussian_opacity,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_range,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_size,
                                          torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_grid_density_tensor,
                                          torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussians_cov_inv,
                                            torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> d_num_voxels_per_gaussian,
                                            const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_lower_boundary,
                                            const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_upper_boundary,
                                            const int voxel_dim_x,
                                            const int voxel_dim_y,
                                            const int voxel_dim_z
                                            ) {
        int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int gaussian_idx = blockIdx.x / num_block_per_gaussian;
        int threads_per_gaussian = num_block_per_gaussian * num_threads_per_block;
        int local_thread_idx = global_thread_idx % threads_per_gaussian;
        int num_gaussians = mean3d.size(0);
        
        if (gaussian_idx >= num_gaussians) return;

        // assume num_voxels_per_gaussian is already set 12160, 12160/4096 + 1 = 3
        // int chunk_size = d_num_voxels_per_gaussian[gaussian_idx] / threads_per_gaussian + 1;
        int chunk_size = (d_num_voxels_per_gaussian[gaussian_idx] + threads_per_gaussian - 1) / threads_per_gaussian;
        // 0*5
        int start_voxel_idx = local_thread_idx * chunk_size;
        // get the end voxel index
        int end_voxel_idx = min(start_voxel_idx + chunk_size, d_num_voxels_per_gaussian[gaussian_idx]);
        
        // if (start_voxel_idx >= d_num_voxels_per_gaussian[gaussian_idx]) return;

        __shared__ float3 shared_mu;
        __shared__ float shared_cov_inv[covariance_dim];
        __shared__ float shared_opacity;
        // __shared__ float3 sigma;
        __shared__ int3 valid_voxel_lower_idx;
        // __shared__ int3 valid_voxel_upper_idx;
        __shared__ int3 valid_voxel_dims;
        
        // for the current Gaussian
        if(threadIdx.x == 0){
            // mean for the current Gaussian
            shared_mu = make_float3(
                mean3d[gaussian_idx][0],
                mean3d[gaussian_idx][1],
                mean3d[gaussian_idx][2]
            );
            // opacity for the current Gaussian
            shared_opacity = d_gaussian_opacity[gaussian_idx];
            
            // invert the covariance matrix
            // invert_3x3_flat(d_gaussian_covariances[gaussian_idx].data(), shared_cov_inv);
            shared_cov_inv[0] = d_gaussians_cov_inv[gaussian_idx][0];
            shared_cov_inv[1] = d_gaussians_cov_inv[gaussian_idx][1];
            shared_cov_inv[2] = d_gaussians_cov_inv[gaussian_idx][2];
            shared_cov_inv[3] = d_gaussians_cov_inv[gaussian_idx][3];
            shared_cov_inv[4] = d_gaussians_cov_inv[gaussian_idx][4];
            shared_cov_inv[5] = d_gaussians_cov_inv[gaussian_idx][5];
            shared_cov_inv[6] = d_gaussians_cov_inv[gaussian_idx][6];
            shared_cov_inv[7] = d_gaussians_cov_inv[gaussian_idx][7];
            shared_cov_inv[8] = d_gaussians_cov_inv[gaussian_idx][8];

            // boundary for the valid voxels that will be affected by the current Gaussian
            valid_voxel_lower_idx.x = d_valid_voxel_lower_boundary[gaussian_idx][0];
            valid_voxel_lower_idx.y = d_valid_voxel_lower_boundary[gaussian_idx][1];
            valid_voxel_lower_idx.z = d_valid_voxel_lower_boundary[gaussian_idx][2];
            // valid_voxel_upper_idx.x = d_valid_voxel_upper_boundary[gaussian_idx][0];
            // valid_voxel_upper_idx.y = d_valid_voxel_upper_boundary[gaussian_idx][1];
            // valid_voxel_upper_idx.z = d_valid_voxel_upper_boundary[gaussian_idx][2];
            
            // // current Gaussian's valid voxel dimensions
            // // int3 diff = valid_voxel_upper_idx - valid_voxel_lower_idx + 1;
            // valid_voxel_upper_idx.x = min(valid_voxel_upper_idx.x, voxel_dim_x - 1);
            // valid_voxel_upper_idx.y = min(valid_voxel_upper_idx.y, voxel_dim_y - 1);
            // valid_voxel_upper_idx.z = min(valid_voxel_upper_idx.z, voxel_dim_z - 1);

            int valid_voxel_dim_x = max(0, d_valid_voxel_upper_boundary[gaussian_idx][0] - d_valid_voxel_lower_boundary[gaussian_idx][0] + 1);
            int valid_voxel_dim_y = max(0, d_valid_voxel_upper_boundary[gaussian_idx][1] - d_valid_voxel_lower_boundary[gaussian_idx][1] + 1);
            int valid_voxel_dim_z = max(0, d_valid_voxel_upper_boundary[gaussian_idx][2] - d_valid_voxel_lower_boundary[gaussian_idx][2] + 1);

            valid_voxel_dims = make_int3(valid_voxel_dim_x, valid_voxel_dim_y, valid_voxel_dim_z);

            // valid_voxel_dims = make_int3(
            //     valid_voxel_upper_idx.x - valid_voxel_lower_idx.x + 1,
            //     valid_voxel_upper_idx.y - valid_voxel_lower_idx.y + 1,
            //     valid_voxel_upper_idx.z - valid_voxel_lower_idx.z + 1
            // );

            // save the inverted covariance matrix for the backward pass
            // d_gaussians_cov_inv[gaussian_idx][0] = shared_cov_inv[0];          // xx
            // d_gaussians_cov_inv[gaussian_idx][1] = shared_cov_inv[4];          // yy
            // d_gaussians_cov_inv[gaussian_idx][2] = shared_cov_inv[8];          // zz
            // d_gaussians_cov_inv[gaussian_idx][3] = shared_cov_inv[1];          // xy
            // d_gaussians_cov_inv[gaussian_idx][4] = shared_cov_inv[2];          // xz
            // d_gaussians_cov_inv[gaussian_idx][5] = shared_cov_inv[5];          // yz
        }
        __syncthreads();

        for(int i = start_voxel_idx; i < end_voxel_idx; ++i){
            int local_x = i / (valid_voxel_dims.y * valid_voxel_dims.z);
            int rem     = i % (valid_voxel_dims.y * valid_voxel_dims.z);
            int local_y = rem / valid_voxel_dims.z;
            int local_z = rem % valid_voxel_dims.z;

            int x = valid_voxel_lower_idx.x + local_x;
            int y = valid_voxel_lower_idx.y + local_y;
            int z = valid_voxel_lower_idx.z + local_z;

            if (x < 0 || x >= voxel_dim_x ||
                y < 0 || y >= voxel_dim_y ||
                z < 0 || z >= voxel_dim_z) {
                continue;
            }

            float3 voxel_center = make_float3(
                d_voxel_range[0] + (x + 0.5f) * d_voxel_size[0],
                d_voxel_range[1] + (y + 0.5f) * d_voxel_size[1],
                d_voxel_range[2] + (z + 0.5f) * d_voxel_size[2]);

            // float3 voxel_center = make_float3(
            //     d_voxel_range[0] + x * d_voxel_size[0],
            //     d_voxel_range[1] + y * d_voxel_size[1],
            //     d_voxel_range[2] + z * d_voxel_size[2]);

            float dx[3] = {voxel_center.x - shared_mu.x,
                        voxel_center.y - shared_mu.y,
                        voxel_center.z - shared_mu.z};

            float maha = 0.0f;
            #pragma unroll
            for (int i = 0; i < 3; ++i)
                #pragma unroll
                for (int j = 0; j < 3; ++j)
                    // maha += dx[i] * shared_cov_inv[i * 3 + j] * dx[j];
                    maha = fmaf(dx[i] * shared_cov_inv[i * 3 + j], dx[j], maha);
            
            float opacity = shared_opacity;
            // float density = opacity * expf(-0.5f * maha);
            float exp_term = __expf(-0.5f * maha);  // Use __expf for better performance on CUDA
            float density = opacity * exp_term;  // Use __expf for better performance on CUDA
            
            atomicAdd(&d_grid_density_tensor[x][y][z], density);
        }
    }

    template <typename scalar_t>
    __global__ void ComputeGaussian2VoxelTileWithFeats(
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mean3d,
                                            // const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussian_covariances,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_gaussian_opacity,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_range,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_size,
                                          torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_grid_density_tensor,
                                          torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussians_cov_inv,
                                            torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> d_num_voxels_per_gaussian,
                                            const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_lower_boundary,
                                            const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_upper_boundary,
                                            const int voxel_dim_x,
                                            const int voxel_dim_y,
                                            const int voxel_dim_z,
                                            const int max_tiles_x,
                                            const int max_tiles_y,
                                            const int max_tiles_z,
                                            const int* __restrict__ active_tiles,
                                            const int* __restrict__ runs_offsets,
                                            const int* __restrict__ d_sorted_gaussian_ids,
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_features,
                                            torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_grid_feats,
                                            const int num_active_tiles,
                                            const int num_runs,
                                            const int num_gaussians
                                            ) {
        // decode the tile to global tile coordinates
        int tile_idx = blockIdx.x;
        if (tile_idx >= num_active_tiles) return;
        const int global_tile_id = active_tiles[tile_idx];
        int tx = global_tile_id / (max_tiles_y * max_tiles_z);
        int rem = global_tile_id % (max_tiles_y * max_tiles_z);
        int ty = rem / max_tiles_z;
        int tz = rem % max_tiles_z;

        // convert threadIdx.x to local voxel index in the tile
        int local_voxel_idx = threadIdx.x;
        if (local_voxel_idx >= TILE_X * TILE_Y * TILE_Z) return;

        int local_voxel_x_idx = local_voxel_idx / (TILE_Y * TILE_Z);
        int rem1 = local_voxel_idx % (TILE_Y * TILE_Z);
        int local_voxel_y_idx = rem1 / TILE_Z;
        int local_voxel_z_idx = rem1 % TILE_Z;

        // convert local voxel index to global voxel index
        const int x = tx * TILE_X + local_voxel_x_idx;
        const int y = ty * TILE_Y + local_voxel_y_idx;
        const int z = tz * TILE_Z + local_voxel_z_idx;

        // if (x >= voxel_dim_x || y >= voxel_dim_y || z >= voxel_dim_z){
        //     #ifdef DEBUG_MODE
        //     printf("Error: voxel (%d,%d,%d) is out of bound (%d,%d,%d)\n", x, y, z, voxel_dim_x, voxel_dim_y, voxel_dim_z);
        //     #endif
        //     return;
        // }

        // calculate the center of the voxel
        float3 voxel_center = make_float3(
            d_voxel_range[0] + (x + 0.5f) * d_voxel_size[0],
            d_voxel_range[1] + (y + 0.5f) * d_voxel_size[1],
            d_voxel_range[2] + (z + 0.5f) * d_voxel_size[2]);

        // get the gaussians that are affecting the current tile
        int start_run = runs_offsets[tile_idx];
        int end_run = runs_offsets[tile_idx + 1];
        // int start_run = runs_offsets[global_tile_id];
        // int end_run   = runs_offsets[global_tile_id + 1];

        // if (tile_idx >= num_active_tiles - 1) {
        //     end_run = num_runs;  // Clamp to maximum valid index
        // }

        __shared__ float share_density[TILE_X * TILE_Y * TILE_Z];

        const bool in_bounds = (x < voxel_dim_x && y < voxel_dim_y && z < voxel_dim_z);

        const bool live = (threadIdx.x < TILE_X * TILE_Y * TILE_Z);

        float dens_acc = 0.f;

        if (live && threadIdx.y == 0) {
            share_density[local_voxel_idx] = 0.0f;
        }
        __syncthreads();

        int feats_dim = d_features.size(1);
        const int C_BLOCK = 32;
        // Each thread y need to calculate the feats_dim / BlockDim.y features.
        // In order to reduce the pressure of register, we set
        for (int c0 = 0; c0 < feats_dim; c0 += C_BLOCK) {
            // each threadIdx.y will execute C_BLOCK / BlockDim.y features channels
            int feats_channel = (C_BLOCK + blockDim.y - 1) / blockDim.y;
            int c_start = c0 + threadIdx.y * feats_channel;
            int c_end = min(c0 + C_BLOCK, c_start + feats_channel);

            // each thread will have its own feature accumulator
            float feat_accum[C_BLOCK] = {0.0f};

            // iterate over all the gaussians that are affecting the current tile
            for (int run_idx = start_run; run_idx < end_run; ++run_idx) {
                // if(run_idx >= num_runs) break;
                // if (run_idx < 0 || run_idx >= num_runs) continue;
                int gaussian_idx = d_sorted_gaussian_ids[run_idx];
                if (gaussian_idx < 0 || gaussian_idx >= num_gaussians) continue;
                if (d_num_voxels_per_gaussian[gaussian_idx] == 0) continue;

                float density = 0.0f;

                // if (live && threadIdx.y == 0) {
                //     share_density[local_voxel_idx] = 0.0f;
                // }
                // __syncthreads();

                if(live && threadIdx.y == 0 && in_bounds){
                    const int lx = d_valid_voxel_lower_boundary[gaussian_idx][0];
                    const int ly = d_valid_voxel_lower_boundary[gaussian_idx][1];
                    const int lz = d_valid_voxel_lower_boundary[gaussian_idx][2];
                    const int ux = d_valid_voxel_upper_boundary[gaussian_idx][0];
                    const int uy = d_valid_voxel_upper_boundary[gaussian_idx][1];
                    const int uz = d_valid_voxel_upper_boundary[gaussian_idx][2];
                    if (!(x < lx || x > ux || y < ly || y > uy || z < lz || z > uz)) {
                        // check if the voxel is within the valid boundary of the gaussian
                        // if () {
                            float3 mu = make_float3(
                                mean3d[gaussian_idx][0],
                                mean3d[gaussian_idx][1],
                                mean3d[gaussian_idx][2]
                            );

                            const float* __restrict__ cov_inv = &d_gaussians_cov_inv[gaussian_idx][0];
                            // const float opacity = ;

                            float dx[3] = {voxel_center.x - mu.x,
                                            voxel_center.y - mu.y,
                                            voxel_center.z - mu.z};

                            // float maha = 0.0f;
                            // #pragma unroll
                            // for (int i = 0; i < 3; ++i)
                            //     #pragma unroll
                            //     for (int j = 0; j < 3; ++j)
                            //         // maha += dx[i] * cov_inv[i * 3 + j] * dx[j];
                            //         maha = fmaf(dx[i] * cov_inv[i * 3 + j], dx[j], maha);
                            float qx = cov_inv[0] * dx[0] + cov_inv[1] * dx[1] + cov_inv[2] * dx[2];
                            float qy = cov_inv[3] * dx[0] + cov_inv[4] * dx[1] + cov_inv[5] * dx[2];
                            float qz = cov_inv[6] * dx[0] + cov_inv[7] * dx[1] + cov_inv[8] * dx[2];
                            float maha = dx[0] * qx + dx[1] * qy + dx[2] * qz;
                            // float qy = cov_inv[3]*dx0 + cov_inv[4]*dx1 + cov_inv[5]*dx2;
                            // float qz = cov_inv[6]*dx0 + cov_inv[7]*dx1 + cov_inv[8]*dx2;
                            // float maha = dx0*qx + dx1*qy + dx2*qz;

                            
                            float exp_term = __expf(-0.5f * maha);  // Use __expf for better performance on CUDA
                            density = d_gaussian_opacity[gaussian_idx] * exp_term;

                            // the density is accumulated only once, not once per channel block
                            if(c0 == 0){
                                dens_acc += density;
                            }
                        // }
                    }
                    share_density[local_voxel_idx] = density;
                }
                __syncthreads();

                const float d = (live ? share_density[local_voxel_idx] : 0.0f);

                // // accumulate the feature channels handled by this block
                if (live && in_bounds && d > 0.f && c_start < feats_dim) {
                    #pragma unroll
                    for (int c = c_start; c < c_end; ++c) {
                        if (c < feats_dim) {
                            feat_accum[c - c_start] = fmaf(d, d_features[gaussian_idx][c],
                                                        feat_accum[c - c_start]);
                        }
                    }
                }

                __syncthreads();
            }

            if (live && in_bounds && c_start < feats_dim) {
                #pragma unroll
                for (int c = c_start; c < c_end; ++c) {
                    if (c < feats_dim) {
                        d_grid_feats[x][y][z][c] += feat_accum[c - c_start];
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.y == 0 && in_bounds && live) {
            d_grid_density_tensor[x][y][z] += dens_acc;
            // if(threadIdx.x == 0 and threadIdx.y == 0){
            //     printf("successfully finish the density computation for tile (%d,%d,%d) voxel (%d,%d,%d), density: %f \n", tx, ty, tz, x, y, z, dens_acc);
            // }
        }
    }

    

    template <typename scalar_t>
    __global__ void ComputeGaussian2VoxelWithFeats(
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mean3d,
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussian_covariances,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_gaussian_opacity,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_range,
                                            const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_size,
                                          torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_grid_density_tensor,
                                          torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussians_cov_inv,
                                            torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> d_num_voxels_per_gaussian,
                                            const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_lower_boundary,
                                            const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_upper_boundary,
                                            const int voxel_dim_x,
                                            const int voxel_dim_y,
                                            const int voxel_dim_z,
                                            const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_features,
                                            torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_grid_feats
                                            ) {
        int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int gaussian_idx = blockIdx.x / num_block_per_gaussian;
        int threads_per_gaussian = num_block_per_gaussian * num_threads_per_block;
        int local_thread_idx = global_thread_idx % threads_per_gaussian;
        int num_gaussians = mean3d.size(0);

        if (gaussian_idx >= num_gaussians) return;
        int num_feature = d_features.size(1);
        // assume num_voxels_per_gaussian is already set 12160, 12160/4096 + 1 = 3
        // int chunk_size = d_num_voxels_per_gaussian[gaussian_idx] / threads_per_gaussian + 1;
        int chunk_size = (d_num_voxels_per_gaussian[gaussian_idx] + threads_per_gaussian - 1) / threads_per_gaussian;
        // 0*5
        int start_voxel_idx = local_thread_idx * chunk_size;
        // get the end voxel index
        int end_voxel_idx = min(start_voxel_idx + chunk_size, d_num_voxels_per_gaussian[gaussian_idx]);
        
        // if (start_voxel_idx >= d_num_voxels_per_gaussian[gaussian_idx]) return;

        __shared__ float3 shared_mu;
        __shared__ float shared_cov_inv[covariance_dim];
        __shared__ float shared_opacity;
        // __shared__ float3 sigma;
        __shared__ int3 valid_voxel_lower_idx;
        // __shared__ int3 valid_voxel_upper_idx;
        __shared__ int3 valid_voxel_dims;
        
        // for the current Gaussian
        if(threadIdx.x == 0){
            // mean for the current Gaussian
            shared_mu = make_float3(
                mean3d[gaussian_idx][0],
                mean3d[gaussian_idx][1],
                mean3d[gaussian_idx][2]
            );
            // opacity for the current Gaussian
            shared_opacity = d_gaussian_opacity[gaussian_idx];
            
            // invert the covariance matrix
            invert_3x3_flat(d_gaussian_covariances[gaussian_idx].data(), shared_cov_inv);

            // boundary for the valid voxels that will be affected by the current Gaussian
            valid_voxel_lower_idx.x = d_valid_voxel_lower_boundary[gaussian_idx][0];
            valid_voxel_lower_idx.y = d_valid_voxel_lower_boundary[gaussian_idx][1];
            valid_voxel_lower_idx.z = d_valid_voxel_lower_boundary[gaussian_idx][2];
            // valid_voxel_upper_idx.x = d_valid_voxel_upper_boundary[gaussian_idx][0];
            // valid_voxel_upper_idx.y = d_valid_voxel_upper_boundary[gaussian_idx][1];
            // valid_voxel_upper_idx.z = d_valid_voxel_upper_boundary[gaussian_idx][2];
            
            // // current Gaussian's valid voxel dimensions
            // // int3 diff = valid_voxel_upper_idx - valid_voxel_lower_idx + 1;
            // valid_voxel_upper_idx.x = min(valid_voxel_upper_idx.x, voxel_dim_x - 1);
            // valid_voxel_upper_idx.y = min(valid_voxel_upper_idx.y, voxel_dim_y - 1);
            // valid_voxel_upper_idx.z = min(valid_voxel_upper_idx.z, voxel_dim_z - 1);

            int valid_voxel_dim_x = max(0, d_valid_voxel_upper_boundary[gaussian_idx][0] - d_valid_voxel_lower_boundary[gaussian_idx][0] + 1);
            int valid_voxel_dim_y = max(0, d_valid_voxel_upper_boundary[gaussian_idx][1] - d_valid_voxel_lower_boundary[gaussian_idx][1] + 1);
            int valid_voxel_dim_z = max(0, d_valid_voxel_upper_boundary[gaussian_idx][2] - d_valid_voxel_lower_boundary[gaussian_idx][2] + 1);

            valid_voxel_dims = make_int3(valid_voxel_dim_x, valid_voxel_dim_y, valid_voxel_dim_z);

            // valid_voxel_dims = make_int3(
            //     valid_voxel_upper_idx.x - valid_voxel_lower_idx.x + 1,
            //     valid_voxel_upper_idx.y - valid_voxel_lower_idx.y + 1,
            //     valid_voxel_upper_idx.z - valid_voxel_lower_idx.z + 1
            // );

            // save the inverted covariance matrix for the backward pass
            d_gaussians_cov_inv[gaussian_idx][0] = shared_cov_inv[0];          // xx
            d_gaussians_cov_inv[gaussian_idx][1] = shared_cov_inv[4];          // yy
            d_gaussians_cov_inv[gaussian_idx][2] = shared_cov_inv[8];          // zz
            d_gaussians_cov_inv[gaussian_idx][3] = shared_cov_inv[1];          // xy
            d_gaussians_cov_inv[gaussian_idx][4] = shared_cov_inv[2];          // xz
            d_gaussians_cov_inv[gaussian_idx][5] = shared_cov_inv[5];          // yz
        }
        __syncthreads();

        for(int i = start_voxel_idx; i < end_voxel_idx; ++i){
            int local_x = i / (valid_voxel_dims.y * valid_voxel_dims.z);
            int rem     = i % (valid_voxel_dims.y * valid_voxel_dims.z);
            int local_y = rem / valid_voxel_dims.z;
            int local_z = rem % valid_voxel_dims.z;

            int x = valid_voxel_lower_idx.x + local_x;
            int y = valid_voxel_lower_idx.y + local_y;
            int z = valid_voxel_lower_idx.z + local_z;

            if (x < 0 || x >= voxel_dim_x ||
                y < 0 || y >= voxel_dim_y ||
                z < 0 || z >= voxel_dim_z) {
                continue;
            }

            float3 voxel_center = make_float3(
                d_voxel_range[0] + (x + 0.5f) * d_voxel_size[0],
                d_voxel_range[1] + (y + 0.5f) * d_voxel_size[1],
                d_voxel_range[2] + (z + 0.5f) * d_voxel_size[2]);

            // float3 voxel_center = make_float3(
            //     d_voxel_range[0] + x * d_voxel_size[0],
            //     d_voxel_range[1] + y * d_voxel_size[1],
            //     d_voxel_range[2] + z * d_voxel_size[2]);

            float dx[3] = {voxel_center.x - shared_mu.x,
                        voxel_center.y - shared_mu.y,
                        voxel_center.z - shared_mu.z};

            float maha = 0.0f;
            #pragma unroll
            for (int i = 0; i < 3; ++i)
                #pragma unroll
                for (int j = 0; j < 3; ++j)
                    // maha += dx[i] * shared_cov_inv[i * 3 + j] * dx[j];
                    maha = fmaf(dx[i] * shared_cov_inv[i * 3 + j], dx[j], maha);
            
            float opacity = shared_opacity;
            // float density = opacity * expf(-0.5f * maha);
            float exp_term = __expf(-0.5f * maha);  // Use __expf for better performance on CUDA
            float density = opacity * exp_term;  // Use __expf for better performance on CUDA
            
            // atomicAdd(&d_grid_density_tensor[x][y][z], density);
            if (blockIdx.y == 0) {
                atomicAdd(&d_grid_density_tensor[x][y][z], density);
            }

            // atomicAdd(&d_grid_feats[x][y][z][blockIdx.y], density * d_features[gaussian_idx][blockIdx.y]);

            for(int i = 0; i < num_feature; ++i){
                atomicAdd(&d_grid_feats[x][y][z][i], density * d_features[gaussian_idx][i]);
            }
        }
    }

    union float2UllUnion {
        float2 f;
        unsigned long long int ull;
    };

    __device__ void atomicAddKahan(float2* __restrict address, const float val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        float2UllUnion old, assumed, tmp;
        old.ull = *address_as_ull;
        do {
            assumed = old;
            tmp = assumed;
            // kahan summation
            const float y = val - tmp.f.y;
            const float t = tmp.f.x + y;
            tmp.f.y = (t - tmp.f.x) - y;
            tmp.f.x = t;

            old.ull = atomicCAS(address_as_ull, assumed.ull, tmp.ull);

        } while (assumed.ull != old.ull);

    }

    __device__ __forceinline__ void matrixMultiply3x3(
        const float* a, const float* b, float* c) {
        c[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6];
        c[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7];
        c[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];
        c[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6];
        c[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7];
        c[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8];
        c[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6];
        c[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7];
        c[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8];
    }

    // __device__ __forceinline__ float warpReduceSum(float val) {
    //     #pragma unroll
    //     for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    //         val += __shfl_down_sync(0xFFFFFFFFU, val, offset);
    //     }
    //     return val;
    // }

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
     * @brief Sum N per-thread values over the whole block. The block size must be a multiple of warp_size, and every thread of the block must call it.
     * @details The totals end up in v[] of the first thread of the block.
     */
    template <int N>
    __device__ __forceinline__ void blockReduceSumArray(float (&v)[N]) {
        __shared__ float warp_sums[N][warp_size];
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int lane_id = tid % warp_size;
        const int warp_id = tid / warp_size;
        const int num_warps = (blockDim.x * blockDim.y) / warp_size;

        #pragma unroll
        for (int k = 0; k < N; ++k) v[k] = warpReduceSum(v[k]);

        if (lane_id == 0) {
            #pragma unroll
            for (int k = 0; k < N; ++k) warp_sums[k][warp_id] = v[k];
        }
        __syncthreads();

        if (warp_id == 0) {
            #pragma unroll
            for (int k = 0; k < N; ++k) v[k] = warpReduceSum((lane_id < num_warps) ? warp_sums[k][lane_id] : 0.0f);
        }
    }

    // Global voxel index of the local_voxel_idx-th voxel of a tile (same layout as ComputeGaussian2VoxelTile)
    __device__ __forceinline__ int3 DecodeTileVoxel(const int tile_id, const int local_voxel_idx, const int max_tiles_y, const int max_tiles_z) {
        const int tx = tile_id / (max_tiles_y * max_tiles_z);
        const int rem = tile_id % (max_tiles_y * max_tiles_z);
        const int ty = rem / max_tiles_z;
        const int tz = rem % max_tiles_z;

        const int rem1 = local_voxel_idx % (TILE_Y * TILE_Z);
        return make_int3(tx * TILE_X + local_voxel_idx / (TILE_Y * TILE_Z),
                         ty * TILE_Y + rem1 / TILE_Z,
                         tz * TILE_Z + rem1 % TILE_Z);
    }

    // exp(-0.5 * maha) with dx = voxel_center - mean, and Adx = cov_inv * dx
    __device__ __forceinline__ float EvalGaussian(const float (&dx)[3], const float (&cov_inv)[covariance_dim], float3& Adx) {
        Adx.x = cov_inv[0] * dx[0] + cov_inv[1] * dx[1] + cov_inv[2] * dx[2];
        Adx.y = cov_inv[3] * dx[0] + cov_inv[4] * dx[1] + cov_inv[5] * dx[2];
        Adx.z = cov_inv[6] * dx[0] + cov_inv[7] * dx[1] + cov_inv[8] * dx[2];
        const float maha = dx[0] * Adx.x + dx[1] * Adx.y + dx[2] * Adx.z;
        return __expf(-0.5f * maha);
    }

    // Layout of the per-Gaussian gradient accumulators: mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz, opacity
    constexpr int num_gaussian_grads = 10;

    /**
     * @brief Add the contribution of one voxel to the gradients of one Gaussian.
     * @param grad_w dL/dw, where w = opacity * exp(-0.5 * maha) is the density of the Gaussian at the voxel
     */
    __device__ __forceinline__ void AccumulateGaussianGrads(float (&grads)[num_gaussian_grads], const float grad_w,
                                                            const float exp_term, const float opacity, const float3 Adx) {
        const float grad_wd = grad_w * opacity * exp_term;

        // dw/dmu = w * cov_inv * (x - mu)
        grads[0] += grad_wd * Adx.x;
        grads[1] += grad_wd * Adx.y;
        grads[2] += grad_wd * Adx.z;

        // dw/dcov = 0.5 * w * cov_inv (x - mu) (x - mu)^T cov_inv
        const float half_grad_wd = 0.5f * grad_wd;
        grads[3] += half_grad_wd * Adx.x * Adx.x;
        grads[4] += half_grad_wd * Adx.y * Adx.y;
        grads[5] += half_grad_wd * Adx.z * Adx.z;
        grads[6] += half_grad_wd * Adx.x * Adx.y;
        grads[7] += half_grad_wd * Adx.x * Adx.z;
        grads[8] += half_grad_wd * Adx.y * Adx.z;

        // dw/dopacity = exp(-0.5 * maha)
        grads[9] += grad_w * exp_term;
    }

    // Write the reduced gradients of one Gaussian. The covariance gradient is symmetric and stored as a flattened 3x3.
    template <typename scalar_t>
    __device__ __forceinline__ void WriteGaussianGrads(const float (&grads)[num_gaussian_grads], const int gaussian_idx,
                                                       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_means,
                                                       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_cov,
                                                       torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_opacity) {
        grad_means[gaussian_idx][0] = grads[0];
        grad_means[gaussian_idx][1] = grads[1];
        grad_means[gaussian_idx][2] = grads[2];

        grad_cov[gaussian_idx][0] = grads[3];   // xx
        grad_cov[gaussian_idx][1] = grads[6];   // xy
        grad_cov[gaussian_idx][2] = grads[7];   // xz
        grad_cov[gaussian_idx][3] = grads[6];   // xy
        grad_cov[gaussian_idx][4] = grads[4];   // yy
        grad_cov[gaussian_idx][5] = grads[8];   // yz
        grad_cov[gaussian_idx][6] = grads[7];   // xz
        grad_cov[gaussian_idx][7] = grads[8];   // yz
        grad_cov[gaussian_idx][8] = grads[5];   // zz

        grad_opacity[gaussian_idx] = grads[9];
    }

    /**
     * @brief Backward of the density based on the Gaussian->Tile CSR.
     * @details One Gaussian per block. The block walks the tiles g2t_tile_ids[g2t_offsets[g] : g2t_offsets[g+1]] of its Gaussian with one voxel
     *          per thread, then reduces the gradients inside the block and writes them directly, so no atomic operation is needed.
     */
    template <typename scalar_t>
    __global__ void ComputeGaussian2VoxelGradientTile(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_density_output,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mean3d,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussians_cov_inv,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_gaussian_opacity,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_range,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_size,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_lower_boundary,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_upper_boundary,
        const int* __restrict__ g2t_offsets,
        const int* __restrict__ g2t_tile_ids,
        const int max_tiles_y,
        const int max_tiles_z,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_means,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_cov,
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_opacity) {
        // one Gaussian per block
        const int gaussian_idx = blockIdx.x;
        if (gaussian_idx >= mean3d.size(0)) return;

        // the tiles touched by the current Gaussian
        constexpr int vox_per_tile = TILE_X * TILE_Y * TILE_Z;
        const int tile_start = g2t_offsets[gaussian_idx];
        const int num_voxels = (g2t_offsets[gaussian_idx + 1] - tile_start) * vox_per_tile;

        const float3 mu = make_float3(mean3d[gaussian_idx][0], mean3d[gaussian_idx][1], mean3d[gaussian_idx][2]);
        const float opacity = d_gaussian_opacity[gaussian_idx];
        float cov_inv[covariance_dim];
        #pragma unroll
        for (int k = 0; k < covariance_dim; ++k) cov_inv[k] = d_gaussians_cov_inv[gaussian_idx][k];

        // the valid voxel boundary of the current Gaussian (already clamped to the grid)
        const int3 lb = make_int3(d_valid_voxel_lower_boundary[gaussian_idx][0], d_valid_voxel_lower_boundary[gaussian_idx][1], d_valid_voxel_lower_boundary[gaussian_idx][2]);
        const int3 ub = make_int3(d_valid_voxel_upper_boundary[gaussian_idx][0], d_valid_voxel_upper_boundary[gaussian_idx][1], d_valid_voxel_upper_boundary[gaussian_idx][2]);

        float grads[num_gaussian_grads] = {0.0f};

        for (int i = threadIdx.x; i < num_voxels; i += blockDim.x) {
            const int3 v = DecodeTileVoxel(g2t_tile_ids[tile_start + i / vox_per_tile], i % vox_per_tile, max_tiles_y, max_tiles_z);

            // only the voxels that the forward splats the Gaussian into
            if (v.x < lb.x || v.x > ub.x || v.y < lb.y || v.y > ub.y || v.z < lb.z || v.z > ub.z) continue;

            const float dx[3] = {d_voxel_range[0] + (v.x + 0.5f) * d_voxel_size[0] - mu.x,
                                 d_voxel_range[1] + (v.y + 0.5f) * d_voxel_size[1] - mu.y,
                                 d_voxel_range[2] + (v.z + 0.5f) * d_voxel_size[2] - mu.z};
            float3 Adx;
            const float exp_term = EvalGaussian(dx, cov_inv, Adx);

            AccumulateGaussianGrads(grads, grad_density_output[v.x][v.y][v.z], exp_term, opacity, Adx);
        }

        // the whole Gaussian is handled by this block, so the result is written directly
        blockReduceSumArray(grads);
        if (threadIdx.x == 0) {
            WriteGaussianGrads(grads, gaussian_idx, grad_means, grad_cov, grad_opacity);
        }
    }

    /**
     * @brief Backward of the density and the features based on the Gaussian->Tile CSR.
     * @details One Gaussian per block, block = (warp_size, warps_per_block). Each warp takes one voxel of the Gaussian's tiles at a time and its
     *          lanes split the feature channels. With w_gv the density of Gaussian g at voxel v, D_v = sum_g w_gv and
     *          F_vc = sum_g w_gv * f_gc / max(D_v, eps):
     *              dL/dw_gv = dL/dD_v + sum_c dL/dF_vc * dF_vc/dw_gv,  dF_vc/dw_gv = (f_gc - F_vc) / D_v   (f_gc / eps if D_v < eps)
     *              dL/df_gc = sum_v dL/dF_vc * w_gv / max(D_v, eps)
     *          sum_c dL/dF_vc * F_vc does not depend on the Gaussian, so it is precomputed once per voxel (d_grad_dot_feats) and the kernel only
     *          reads dL/dF_v for each Gaussian-voxel pair.
     *          The feature gradients are accumulated in one shared memory row per warp (dynamic shared memory: warps_per_block * feats_dim floats),
     *          and the rows are summed at the end, so no atomic operation is needed.
     */
    template <typename scalar_t>
    __global__ void ComputeGaussian2VoxelGradientTileWithFeats(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_density_output,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_feats_output,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mean3d,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gaussians_cov_inv,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_gaussian_opacity,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_grid_density,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_feature,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_grad_dot_feats,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_range,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_voxel_size,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_lower_boundary,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> d_valid_voxel_upper_boundary,
        const int* __restrict__ g2t_offsets,
        const int* __restrict__ g2t_tile_ids,
        const int max_tiles_y,
        const int max_tiles_z,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_means,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_cov,
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_opacity,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_feats) {
        // one Gaussian per block
        const int gaussian_idx = blockIdx.x;
        if (gaussian_idx >= mean3d.size(0)) return;

        const int lane = threadIdx.x;               // feature channel
        const int warp_in_block = threadIdx.y;      // voxel
        const int warps_per_block = blockDim.y;
        const int feats_dim = d_feature.size(1);

        // each warp only touches its own row
        extern __shared__ float sm_grad_feats[];    // [warps_per_block][feats_dim]
        float* warp_grad_feats = sm_grad_feats + warp_in_block * feats_dim;
        for (int c = lane; c < feats_dim; c += warp_size) warp_grad_feats[c] = 0.0f;

        // the tiles touched by the current Gaussian
        constexpr int vox_per_tile = TILE_X * TILE_Y * TILE_Z;
        const int tile_start = g2t_offsets[gaussian_idx];
        const int num_voxels = (g2t_offsets[gaussian_idx + 1] - tile_start) * vox_per_tile;

        const float3 mu = make_float3(mean3d[gaussian_idx][0], mean3d[gaussian_idx][1], mean3d[gaussian_idx][2]);
        const float opacity = d_gaussian_opacity[gaussian_idx];
        float cov_inv[covariance_dim];
        #pragma unroll
        for (int k = 0; k < covariance_dim; ++k) cov_inv[k] = d_gaussians_cov_inv[gaussian_idx][k];

        // the valid voxel boundary of the current Gaussian (already clamped to the grid)
        const int3 lb = make_int3(d_valid_voxel_lower_boundary[gaussian_idx][0], d_valid_voxel_lower_boundary[gaussian_idx][1], d_valid_voxel_lower_boundary[gaussian_idx][2]);
        const int3 ub = make_int3(d_valid_voxel_upper_boundary[gaussian_idx][0], d_valid_voxel_upper_boundary[gaussian_idx][1], d_valid_voxel_upper_boundary[gaussian_idx][2]);

        float grads[num_gaussian_grads] = {0.0f};

        // the voxel and the branches below are the same for all the lanes of a warp, so the warp shuffles are safe
        for (int i = warp_in_block; i < num_voxels; i += warps_per_block) {
            const int3 v = DecodeTileVoxel(g2t_tile_ids[tile_start + i / vox_per_tile], i % vox_per_tile, max_tiles_y, max_tiles_z);

            // only the voxels that the forward splats the Gaussian into
            if (v.x < lb.x || v.x > ub.x || v.y < lb.y || v.y > ub.y || v.z < lb.z || v.z > ub.z) continue;

            const float dx[3] = {d_voxel_range[0] + (v.x + 0.5f) * d_voxel_size[0] - mu.x,
                                 d_voxel_range[1] + (v.y + 0.5f) * d_voxel_size[1] - mu.y,
                                 d_voxel_range[2] + (v.z + 0.5f) * d_voxel_size[2] - mu.z};
            float3 Adx;
            const float exp_term = EvalGaussian(dx, cov_inv, Adx);
            const float density = opacity * exp_term;

            // the forward divides by max(D_v, eps), the clamp only passes the gradient when D_v >= eps
            const float D = d_grid_density[v.x][v.y][v.z];
            const bool activate = D >= eps;
            const float inv_denom = 1.0f / fmaxf(D, eps);
            const float density_inv_denom = density * inv_denom;
            // loaded before the channel loop to hide the latency
            const float grad_density = grad_density_output[v.x][v.y][v.z];
            const float grad_dot_grid_feat = activate ? d_grad_dot_feats[v.x][v.y][v.z] : 0.0f;

            // sum_c dL/dF_vc * f_gc
            float grad_dot_feat = 0.0f;
            for (int c = lane; c < feats_dim; c += warp_size) {
                const float grad_feat = grad_feats_output[v.x][v.y][v.z][c];
                grad_dot_feat = fmaf(grad_feat, d_feature[gaussian_idx][c], grad_dot_feat);

                warp_grad_feats[c] = fmaf(grad_feat, density_inv_denom, warp_grad_feats[c]);
            }
            grad_dot_feat = warpReduceSum(grad_dot_feat);

            if (lane == 0) {
                // sum_c dL/dF_vc * dF_vc/dw_gv = (sum_c dL/dF_vc * f_gc - sum_c dL/dF_vc * F_vc) / D_v   (the second term is 0 if D_v < eps)
                const float s_gv = (grad_dot_feat - grad_dot_grid_feat) * inv_denom;
                AccumulateGaussianGrads(grads, grad_density + s_gv, exp_term, opacity, Adx);
            }
        }
        __syncthreads();

        // the whole Gaussian is handled by this block, so the results are written directly
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int c = tid; c < feats_dim; c += blockDim.x * blockDim.y) {
            float sum = 0.0f;
            for (int w = 0; w < warps_per_block; ++w) sum += sm_grad_feats[w * feats_dim + c];
            grad_feats[gaussian_idx][c] = sum;
        }

        blockReduceSumArray(grads);
        if (tid == 0) {
            WriteGaussianGrads(grads, gaussian_idx, grad_means, grad_cov, grad_opacity);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float> cuda_gaussian_voxelizer_forward(const torch::Tensor& means3D,
                                    // const torch::Tensor& semantics, 
                                    const torch::Tensor& opacity,
                                    // const torch::Tensor& scale, 
                                    const torch::Tensor& covariances,
                                    const torch::Tensor& voxel_size,
                                    const torch::Tensor& voxel_range,
                                    const c10::optional<torch::Tensor> features
                                    ) {
        /**
         * @brief Forward pass for Gaussian voxelization.
         * @param means3D: Tensor containing the 3D means of the Gaussians. Shape: (N, 3)
         * @param semantics: Tensor containing the semantic labels for each Gaussian. Shape: (N, 1)
         * @param opacity: Tensor containing the opacity values for each Gaussian. Shape: (N, 1)
         * @param scale: Tensor containing the scale values for each Gaussian. Shape: (N, 3)
         * @param covariances: Tensor containing the covariance matrices for each Gaussian. Shape: (N, 3, 3)
         * @return density, cov_inv, valid voxel lower/upper boundary, number of voxels per Gaussian, grid features,
         *         the Gaussian->Tile CSR (index pointers g2t_offsets (N+1), tile ids g2t_tile_ids) used by the backward, and the forward time
         */
        #ifdef NVTX_ENABLED
        NVTX_MARK("ComputeForward Launch");
        NVTX_PUSH("ComputeForward");
        #endif
        // TORCH_CHECK(means3D.scalar_type()==at::kFloat && means3D.is_contiguous() && means3D.is_cuda());
        // TORCH_CHECK(opacity.scalar_type()==at::kFloat && opacity.is_contiguous() && opacity.is_cuda());
        // TORCH_CHECK(covariances.scalar_type()==at::kFloat && covariances.is_contiguous() && covariances.is_cuda());

        int num_gaussians = means3D.size(0);
        int voxel_dim_x = static_cast<int>((voxel_range[3].item<float>() - voxel_range[0].item<float>()) / voxel_size[0].item<float>());
        int voxel_dim_y = static_cast<int>((voxel_range[4].item<float>() - voxel_range[1].item<float>()) / voxel_size[1].item<float>());
        int voxel_dim_z = static_cast<int>((voxel_range[5].item<float>() - voxel_range[2].item<float>()) / voxel_size[2].item<float>());

        // tiles size
        int max_tiles_x = (voxel_dim_x + TILE_X - 1) / TILE_X;
        int max_tiles_y = (voxel_dim_y + TILE_Y - 1) / TILE_Y;
        int max_tiles_z = (voxel_dim_z + TILE_Z - 1) / TILE_Z;
        
        #ifdef DEBUG_MODE
        std::cout<<"check title size: x:"<<max_tiles_x<<"y: "<< max_tiles_y <<"z: "<< max_tiles_z<<std::endl;
        #endif
        auto s = at::cuda::getCurrentCUDAStream();
        // at::cuda::CUDAStreamGuard guard(s);
        
        auto float_opts = means3D.options().dtype(torch::kFloat32);
        auto int_opts = means3D.options().dtype(torch::kInt32);
        auto uint8_opts = means3D.options().dtype(torch::kUInt8);
        torch::Tensor d_grid_density_tensor = torch::full({voxel_dim_x, voxel_dim_y, voxel_dim_z}, 0.0, float_opts);
        torch::Tensor d_gaussians_cov_inv = torch::full({num_gaussians, covariance_dim}, 0.0, float_opts);

        torch::Tensor d_valid_voxel_lower_boundary = torch::zeros({num_gaussians, 3}, int_opts);
        torch::Tensor d_valid_voxel_upper_boundary = torch::zeros({num_gaussians, 3}, int_opts);
        torch::Tensor d_num_voxels_per_gaussian = torch::zeros({num_gaussians}, int_opts);
        
        // tile information
        torch::Tensor d_tile_lower_boundary = torch::empty({num_gaussians, 3}, int_opts);
        torch::Tensor d_tile_higher_boundary = torch::empty({num_gaussians, 3}, int_opts);
        torch::Tensor d_num_tiles_per_gaussian = torch::zeros({num_gaussians}, int_opts);
        // record the Index Pointers of the CSR format
        torch::Tensor d_g2t_offsets = torch::empty({num_gaussians + 1}, int_opts);
        cudaMemset(d_g2t_offsets.data_ptr<int>(), 0, sizeof(int));
        // CHECK_CUDA(cudaMemsetAsync(d_g2t_offsets.data_ptr<int>(), 0, sizeof(int), s.stream()));


        torch::Tensor d_grid_feats = torch::empty({voxel_dim_x, voxel_dim_y, voxel_dim_z, features.has_value() ? features.value().size(1) : 0}, float_opts);
        // std::cout << "d_grid_feats size: " << d_grid_feats.sizes() << std::endl;
        #ifdef CUDA_TIMER_ENABLED
        cudaEvent_t cuda_start, cuda_stop;
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);

        cudaEventRecord(cuda_start, s.stream());
        #endif

        // Launch the kernel to get the pairs of voxel index and Gaussian index
        int threads_per_block = num_threads_per_block;
        int block_num = (num_gaussians + threads_per_block - 1) / threads_per_block;

        #ifdef NVTX_ENABLED
        NVTX_PUSH("ComputeGaussianVoxelPair");
        #endif
        // auto s = at::cuda::getCurrentCUDAStream();
        // ComputeGaussianVoxelPair<float><<<block_num, threads_per_block>>>(means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        //                                                                 covariances.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        //                                                                 voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        //                                                                 voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        //                                                                 voxel_dim_x,
        //                                                                 voxel_dim_y,
        //                                                                 voxel_dim_z,
        //                                                                 d_num_voxels_per_gaussian.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        //                                                                 d_valid_voxel_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        //                                                                 d_valid_voxel_upper_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>());
        ComputeGaussianVoxelPairTile<float><<<block_num, threads_per_block, 0, s.stream()>>>(means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                        covariances.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                        voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                                                        voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                                                        voxel_dim_x,
                                                                        voxel_dim_y,
                                                                        voxel_dim_z,
                                                                        d_num_voxels_per_gaussian.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                        d_valid_voxel_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                        d_valid_voxel_upper_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                        d_tile_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                        d_tile_higher_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                        d_num_tiles_per_gaussian.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                        d_gaussians_cov_inv.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                        max_tiles_x, max_tiles_y, max_tiles_z
                                                                        );
        #ifdef NVTX_ENABLED
        NVTX_POP(); // End of ComputeGaussianVoxelPair
        #endif
        
        #ifdef DEBUG_MODE
        // d_num_tiles_per_gaussian
        printf("d_num_tiles_per_gaussian before inclusive sum: \n");
        for(int i = 0; i < min(10, num_gaussians); i++){
            printf("%d ", d_num_tiles_per_gaussian[i].item<int>());
        }
        printf("\n");
        #endif

        // size_t temp_bytes = 0;
        size_t scan_temp_bytes = 0;
        // void*  d_temp = nullptr;
        // CSR format to record the offsets of each Gaussian
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(nullptr, scan_temp_bytes, d_num_tiles_per_gaussian.data_ptr<int>(), d_g2t_offsets.data_ptr<int>() + 1, num_gaussians, s.stream()));

        auto temp_storage = torch::empty({static_cast<long long>(scan_temp_bytes)}, uint8_opts);

        // Based on the inclusive sum to get the index pointers (d_g2t_offsets) of the CSR format
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(temp_storage.data_ptr(), scan_temp_bytes, d_num_tiles_per_gaussian.data_ptr<int>(), d_g2t_offsets.data_ptr<int>()+1, num_gaussians, s.stream()));

        CHECK_CUDA(cudaMemsetAsync(d_g2t_offsets.data_ptr<int>(), 0, sizeof(int), s.stream()));
        // CHECK_CUDA(cudaMemset(d_g2t_offsets.data_ptr<int>(), 0, sizeof(int)));
        // d_num_tiles_per_gaussian
        // printf("d_num_tiles_per_gaussian after inclusive sum: \n");
        // for(int i = 0; i < min(10, num_gaussians); i++){
        //     printf("%d ", d_g2t_offsets[i].item<int>());
        // }
        // printf("\n");

        // calculate the number of total tiles that has been effected by all Gaussian
        int total_tiles_touched;
        // offset the pointer to the last element
        // cudaMemcpy(&total_tiles_touched, d_g2t_offsets.data_ptr<int>() + num_gaussians, sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA(cudaMemcpyAsync(&total_tiles_touched,
                d_g2t_offsets.data_ptr<int>() + num_gaussians,
                sizeof(int), cudaMemcpyDeviceToHost, s.stream()));
        CHECK_CUDA(cudaStreamSynchronize(s.stream()));

        // key
        torch::Tensor d_g2t_tile_ids = torch::empty({total_tiles_touched}, int_opts);
        // value
        torch::Tensor d_g2t_gaussian_ids = torch::empty({total_tiles_touched}, int_opts);

        torch::Tensor d_sorted_tile_ids = torch::empty({total_tiles_touched}, int_opts);
        torch::Tensor d_sorted_gaussian_ids = torch::empty({total_tiles_touched}, int_opts);

        // cudaMemset(d_g2t_tile_ids.data_ptr<int>(), 0xFF, total_tiles_touched * sizeof(int));
        // cudaMemset(d_g2t_gaussian_ids.data_ptr<int>(), 0xFF, total_tiles_touched * sizeof(int));

        // parallel compute the tile id for each 
        FillG2TKernel<<<num_gaussians, num_threads_per_block, 0, s.stream()>>>(
                                            // d_tile_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                            // d_tile_higher_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                            d_tile_lower_boundary.data_ptr<int>(),
                                            d_tile_higher_boundary.data_ptr<int>(),
                                            d_g2t_offsets.data_ptr<int>(),
                                            max_tiles_x, max_tiles_y, max_tiles_z, num_gaussians,
                                            d_g2t_tile_ids.data_ptr<int>(),
                                            d_g2t_gaussian_ids.data_ptr<int>());

        // torch::Tensor d_written_counts = torch::zeros({num_gaussians}, int_opts);
        // FillG2TKernel<<<num_gaussians, num_threads_per_block, 0, s.stream()>>>(
        //     d_tile_lower_boundary.data_ptr<int>(),
        //     d_tile_higher_boundary.data_ptr<int>(),
        //     d_g2t_offsets.data_ptr<int>(),
        //     max_tiles_x, max_tiles_y, max_tiles_z, num_gaussians,
        //     d_g2t_tile_ids.data_ptr<int>(),
        //     d_g2t_gaussian_ids.data_ptr<int>(),
        //     d_written_counts.data_ptr<int>());
        // cudaDeviceSynchronize();
        #ifdef DEBUG_MODE
        printf("check d_g2t_tile_ids%d\n", total_tiles_touched);
        for(int i = 0; i < min(10, total_tiles_touched); ++i){
            printf("%d ", d_g2t_tile_ids[i].item<int>());
        }
        #endif

        // size_t temp_bytes = 0;
        size_t sort_temp_bytes = 0;
        // sorting groups all the Gaussians that hit the same tile together
        cub::DeviceRadixSort::SortPairs(
                                        nullptr, sort_temp_bytes,
                                        d_g2t_tile_ids.data_ptr<int>(), d_sorted_tile_ids.data_ptr<int>(),
                                        d_g2t_gaussian_ids.data_ptr<int>(), d_sorted_gaussian_ids.data_ptr<int>(),
                                        total_tiles_touched, 0, 8 * sizeof(int), s.stream());
        // void* d_temp = nullptr;
        // // cudaMalloc(&d_temp, temp_bytes);
        // if (temp_bytes > 0) {
        //     cudaMalloc(&d_temp, temp_bytes);
        // }
        torch::Tensor workspace = torch::empty({static_cast<long long>(sort_temp_bytes)}, uint8_opts);
        void* d_temp = workspace.data_ptr();
        cub::DeviceRadixSort::SortPairs(
                                        d_temp, sort_temp_bytes,
                                        d_g2t_tile_ids.data_ptr<int>(), d_sorted_tile_ids.data_ptr<int>(),
                                        d_g2t_gaussian_ids.data_ptr<int>(), d_sorted_gaussian_ids.data_ptr<int>(),
                                        total_tiles_touched, 0, 8 * sizeof(int), s.stream());
        
        
        #ifdef DEBUG_MODE
        cudaDeviceSynchronize();
        printf("check sorting key: %d\n", total_tiles_touched);
        for(int i = 0; i < min(10, total_tiles_touched); ++i){
            printf("%d ", d_sorted_tile_ids[i].item<int>());
        }
        

        printf("check sorting value: %d\n", total_tiles_touched);
        for(int i = 0; i < min(10, total_tiles_touched); ++i){
            printf("%d ", d_sorted_gaussian_ids[i].item<int>());
        }
        #endif

        int  unique_tiles_count = 0;  // unique tiles count
        const int num_tiles = max_tiles_x * max_tiles_y * max_tiles_z;
        size_t rle_bytes = 0;
        torch::Tensor d_unique_tiles   = torch::empty({num_tiles}, int_opts); // record the unique tile id
        torch::Tensor d_counts_compact = torch::empty({num_tiles}, int_opts); // record the counts for each unique tile id
        torch::Tensor d_num_runs       = torch::empty({1}, int_opts); // record the number of unique tiles
        // compress the sorted tile ids into unique tile ids and their counts
        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRunLengthEncode.html
        cub::DeviceRunLengthEncode::Encode(
            nullptr, rle_bytes,
            d_sorted_tile_ids.data_ptr<int>(),                   // in  keys (tile_id sorted)
            d_unique_tiles.data_ptr<int>(),               // out unique keys (length >= M)
            d_counts_compact.data_ptr<int>(),             // out counts        (length >= M)
            d_num_runs.data_ptr<int>(),                   // out U on device (int*)
            total_tiles_touched,
            s.stream());              
        auto rle_temp = torch::empty({(long long)rle_bytes}, uint8_opts);
        cub::DeviceRunLengthEncode::Encode(
            rle_temp.data_ptr(), rle_bytes,
            d_sorted_tile_ids.data_ptr<int>(),
            d_unique_tiles.data_ptr<int>(), d_counts_compact.data_ptr<int>(), d_num_runs.data_ptr<int>(), total_tiles_touched, s.stream());

        // get the number of unique tiles
        CHECK_CUDA(cudaMemcpyAsync(&unique_tiles_count, d_num_runs.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, s.stream()));
        CHECK_CUDA(cudaStreamSynchronize(s.stream()));
        // cudaMemcpy(&unique_tiles_count,
        //         d_num_runs.data_ptr<int>(),
        //         sizeof(int), cudaMemcpyDeviceToHost);

        // int num_tiles = max_tiles_x * max_tiles_y * max_tiles_z;
        
        // We just need the activate tile, so we don't need blow code
        // // record each tile's counts of the number of Gaussians that have effect on it
        // torch::Tensor t2g_counts  = torch::zeros({num_tiles}, int_opts);

        // // scatter: write (unique_tiles[i], counts_compact[i]) to t2g_counts[ tile ]
        
        // ScatterCounts<<<(unique_tiles_count+255)/256,256>>>(d_unique_tiles.data_ptr<int>(), d_counts_compact.data_ptr<int>(), unique_tiles_count, t2g_counts.data_ptr<int>());
        
        // // t2g_offsets[t] gives the start index in d_sorted_gaussian_ids for the list of Gaussians that affect tile t.
        // // Interval: [ t2g_offsets[t], t2g_offsets[t+1] )
        // torch::Tensor t2g_offsets = torch::empty({num_tiles + 1}, int_opts);
        // // prefix sum to get the offsets
        // size_t scan_bytes=0;
        // cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,t2g_counts.data_ptr<int>(), t2g_offsets.data_ptr<int>() + 1, num_tiles);
        // auto scan_tmp = torch::empty({(long long)scan_bytes}, int_opts);
        // cub::DeviceScan::InclusiveSum(scan_tmp.data_ptr(), scan_bytes, t2g_counts.data_ptr<int>(), t2g_offsets.data_ptr<int>() + 1, num_tiles);

        // cudaMemset(t2g_offsets.data_ptr<int>(), 0, sizeof(int));

        torch::Tensor runs_offsets = torch::empty({unique_tiles_count + 1}, int_opts);

        // InclusiveSum into [1:], then set [0] to 0 (equivalent to an exclusive sum)
        size_t bytes = 0;
        // auto stream = at::cuda::getCurrentCUDAStream();
        cub::DeviceScan::InclusiveSum(nullptr, bytes,
            d_counts_compact.data_ptr<int>(),
            runs_offsets.data_ptr<int>() + 1,
            unique_tiles_count, s.stream());
        auto tmp = torch::empty({(long long)bytes}, uint8_opts);
        cub::DeviceScan::InclusiveSum(tmp.data_ptr(), bytes,
            d_counts_compact.data_ptr<int>(),
            runs_offsets.data_ptr<int>() + 1,
            unique_tiles_count, s.stream());
        CHECK_CUDA(cudaMemsetAsync(runs_offsets.data_ptr<int>(), 0, sizeof(int), s.stream()));
        // CHECK_CUDA(cudaMemset(runs_offsets.data_ptr<int>(), 0, sizeof(int)));
        // runs_offsets[U] equals M automatically


        #ifdef NVTX_ENABLED
        NVTX_PUSH("ComputeGaussian2Voxel");
        #endif
        // ComputeGaussian2Voxel<float><<<num_block_per_gaussian * num_gaussians, num_threads_per_block, 0, s.stream()>>>(
        //                                                                                 means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        //                                                                                 covariances.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        //                                                                                 opacity.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        //                                                                                 voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        //                                                                                 voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        //                                                                                 d_grid_density_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        //                                                                                 d_gaussians_cov_inv.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        //                                                                                 d_num_voxels_per_gaussian.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        //                                                                                 d_valid_voxel_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        //                                                                                 d_valid_voxel_upper_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>()
        //                                                                                 );
        if(features.has_value()){
            d_grid_feats.zero_();
            const at::Tensor& d_features = features.value();
            // dim3 block(num_threads_per_block, 1, 1);
            // dim3 grid(num_block_per_gaussian * num_gaussians, 1, 1);
            // ComputeGaussian2VoxelWithFeats<float><<<grid, block>>>(
            //                                                         means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //                                                         covariances.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //                                                         opacity.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            //                                                         voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            //                                                         voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            //                                                         d_grid_density_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            //                                                         d_gaussians_cov_inv.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //                                                         d_num_voxels_per_gaussian.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            //                                                         d_valid_voxel_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            //                                                         d_valid_voxel_upper_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            //                                                         voxel_dim_x,
            //                                                         voxel_dim_y,
            //                                                         voxel_dim_z,
            //                                                         d_features.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //                                                         d_grid_feats.packed_accessor32<float,4,torch::RestrictPtrTraits>()
            //                                                         );
            #ifdef DEBUG_MODE
            std::cout<<"unique_tiles_count: "<<unique_tiles_count<<std::endl;
            #endif
            const int vox_per_tile = TILE_X * TILE_Y * TILE_Z;
            const int by = max(1, 1024 / vox_per_tile);
            // std::cout<<"check by:"<<by<<std::endl;
            dim3 block(vox_per_tile, 1, 1);
            // int feature_block = 32;
            dim3 grid(unique_tiles_count, 1, 1);
            ComputeGaussian2VoxelTileWithFeats<float><<<grid, block, 0, s.stream()>>>(means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                                        // covariances.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                                        opacity.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                                                                        voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                                                                        voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                                                                        d_grid_density_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                                                                                        d_gaussians_cov_inv.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                                        d_num_voxels_per_gaussian.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                                        d_valid_voxel_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                                        d_valid_voxel_upper_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                                        voxel_dim_x,
                                                                                        voxel_dim_y,
                                                                                        voxel_dim_z,
                                                                                        max_tiles_x,
                                                                                        max_tiles_y,
                                                                                        max_tiles_z,
                                                                                        d_unique_tiles.data_ptr<int>(),
                                                                                        runs_offsets.data_ptr<int>(),
                                                                                        d_sorted_gaussian_ids.data_ptr<int>(),
                                                                                        d_features.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                                        d_grid_feats.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
                                                                                        unique_tiles_count,
                                                                                        total_tiles_touched,
                                                                                        num_gaussians
                                                                                );
            auto denom = d_grid_density_tensor.unsqueeze(-1).clamp_min(eps); // [X,Y,Z,1]
            d_grid_feats.div_(denom); // in-place
        }
        else{
            // ComputeGaussian2Voxel<float><<<num_block_per_gaussian * num_gaussians, num_threads_per_block>>>(
            //                                                                             means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //                                                                             // covariances.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //                                                                             opacity.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            //                                                                             voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            //                                                                             voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            //                                                                             d_grid_density_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            //                                                                             d_gaussians_cov_inv.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //                                                                             d_num_voxels_per_gaussian.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            //                                                                             d_valid_voxel_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            //                                                                             d_valid_voxel_upper_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            //                                                                             voxel_dim_x,
            //                                                                             voxel_dim_y,
            //                                                                             voxel_dim_z
            //                                                                             );
            ComputeGaussian2VoxelTile<float><<<unique_tiles_count, TILE_X * TILE_Y * TILE_Z, 0, s.stream()>>>(
                                                                                        means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                                        // covariances.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                                        opacity.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                                                                        voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                                                                        voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                                                                        d_grid_density_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                                                                                        d_gaussians_cov_inv.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                                                                        d_num_voxels_per_gaussian.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                                        d_valid_voxel_lower_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                                        d_valid_voxel_upper_boundary.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                                        voxel_dim_x,
                                                                                        voxel_dim_y,
                                                                                        voxel_dim_z,
                                                                                        max_tiles_x,
                                                                                        max_tiles_y,
                                                                                        max_tiles_z,
                                                                                        d_unique_tiles.data_ptr<int>(),
                                                                                        runs_offsets.data_ptr<int>(),
                                                                                        d_sorted_gaussian_ids.data_ptr<int>(),
                                                                                        unique_tiles_count,
                                                                                        total_tiles_touched,
                                                                                        num_gaussians
                                                                                        );
        }
        #ifdef NVTX_ENABLED
        NVTX_POP(); // End of ComputeGaussian2Voxel
        #endif

        // CHECK_CUDA(cudaStreamSynchronize(s.stream())); // make sure the work on this stream is finished
        // if (d_temp) {
        //     CHECK_CUDA(cudaFree(d_temp));
        // }


        #ifdef NVTX_ENABLED
        NVTX_POP(); // End of ComputeForward Launch
        #endif
        #ifdef CUDA_TIMER_ENABLED
        cudaEventRecord(cuda_stop, s.stream());
        cudaEventSynchronize(cuda_stop);
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, cuda_start, cuda_stop);
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
        // std::cout << "Forward GPU Time: " << gpu_time << " ms" << std::endl;
        return std::make_tuple(d_grid_density_tensor, d_gaussians_cov_inv, d_valid_voxel_lower_boundary, d_valid_voxel_upper_boundary, d_num_voxels_per_gaussian, d_grid_feats, d_g2t_offsets, d_g2t_tile_ids, gpu_time / 1000.0f);
        #else
        return std::make_tuple(d_grid_density_tensor, d_gaussians_cov_inv, d_valid_voxel_lower_boundary, d_valid_voxel_upper_boundary, d_num_voxels_per_gaussian, d_grid_feats, d_g2t_offsets, d_g2t_tile_ids, 0.0f);
        #endif
        
    }

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
                                                                                            const c10::optional<torch::Tensor> features
                                                                                            ) {
        /**
         * @brief Backward pass for Gaussian voxelization, based on the Gaussian->Tile CSR (g2t_offsets, g2t_tile_ids) built by the forward.
         *        One Gaussian per block, so the gradients of each Gaussian are written without atomic operations.
         */

        TORCH_CHECK(grad_density_output.scalar_type()==at::kFloat && grad_density_output.is_contiguous() && grad_density_output.is_cuda());
        TORCH_CHECK(means3D.scalar_type()==at::kFloat && means3D.is_contiguous() && means3D.is_cuda());
        TORCH_CHECK(opacity.scalar_type()==at::kFloat && opacity.is_contiguous() && opacity.is_cuda());
        TORCH_CHECK(cov_inv.scalar_type()==at::kFloat && cov_inv.is_contiguous() && cov_inv.is_cuda());
        TORCH_CHECK(voxel_size.scalar_type()==at::kFloat && voxel_size.is_contiguous() && voxel_size.is_cuda());
        TORCH_CHECK(voxel_range.scalar_type()==at::kFloat && voxel_range.is_contiguous() && voxel_range.is_cuda());
        TORCH_CHECK(valid_voxel_lb.scalar_type()==at::kInt && valid_voxel_lb.is_contiguous() && valid_voxel_lb.is_cuda());
        TORCH_CHECK(valid_voxel_ub.scalar_type()==at::kInt && valid_voxel_ub.is_contiguous() && valid_voxel_ub.is_cuda());
        TORCH_CHECK(num_voxels_per_gaussian.scalar_type()==at::kInt && num_voxels_per_gaussian.is_contiguous() && num_voxels_per_gaussian.is_cuda());
        TORCH_CHECK(g2t_offsets.scalar_type()==at::kInt && g2t_offsets.is_contiguous() && g2t_offsets.is_cuda() && g2t_offsets.numel() == means3D.size(0) + 1);
        TORCH_CHECK(g2t_tile_ids.scalar_type()==at::kInt && g2t_tile_ids.is_contiguous() && g2t_tile_ids.is_cuda());
        // TORCH_CHECK(covariances.scalar_type()==at::kFloat && covariances.is_contiguous() && covariances.is_cuda());
        TORCH_CHECK(density.scalar_type()==at::kFloat && density.is_contiguous() && density.is_cuda());
        TORCH_CHECK(grid_feats.scalar_type()==at::kFloat && grid_feats.is_contiguous() && grid_feats.is_cuda());

        int num_gaussians = means3D.size(0);
        auto float_opts = means3D.options().dtype(torch::kFloat32);

        // tile grid of the forward, used to decode the tile ids of the Gaussian->Tile CSR
        const int max_tiles_y = (grad_density_output.size(1) + TILE_Y - 1) / TILE_Y;
        const int max_tiles_z = (grad_density_output.size(2) + TILE_Z - 1) / TILE_Z;

        // every block writes all the gradients of its Gaussian
        torch::Tensor d_means_grad = torch::full({num_gaussians, dim3d}, 0.0, float_opts);
        torch::Tensor d_cov_grad = torch::full({num_gaussians, 9}, 0.0, float_opts);
        torch::Tensor d_opacity_grad = torch::full({num_gaussians}, 0.0, float_opts);
        torch::Tensor d_grid_feats_grad = torch::full({num_gaussians, features.has_value() ? features.value().size(1) : 0}, 0.0, float_opts);
        // std::cout<< "d_grid_feats_grad shape: " << d_grid_feats_grad.sizes() << std::endl;
        // std::cout<< "grad_feats_output shape: " << grad_feats_output.sizes() << std::endl;
        // std::cout<< "grid_feats shape: " << grid_feats.sizes() << std::endl;
        // std::cout<< "density shape: " << density.sizes() << std::endl;
        // std::cout << "features.shape: " << (features.has_value() ? features.value().sizes() : "None") << std::endl;

        auto s = at::cuda::getCurrentCUDAStream();

        #ifdef CUDA_TIMER_ENABLED
        cudaEvent_t cuda_start, cuda_stop;
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);

        cudaEventRecord(cuda_start, s.stream());
        #endif

        if (features.has_value()){
            const at::Tensor& d_features = features.value();
            const int feats_dim = d_features.size(1);

            // one row of feature gradients per warp in the shared memory, use fewer warps if the feature dimension is large
            const size_t max_shared_bytes = 48 * 1024;
            const int warps_per_block = std::min<int>(num_warps_per_block_backward, max_shared_bytes / (std::max(feats_dim, 1) * sizeof(float)));
            TORCH_CHECK(warps_per_block >= 1, "feature dimension ", feats_dim, " is too large for the shared memory of the backward");
            const size_t shared_bytes = warps_per_block * feats_dim * sizeof(float);

            // sum_c dL/dF_vc * F_vc for each voxel, shared by all the Gaussians touching the voxel
            const int64_t num_voxels = density.numel();
            torch::Tensor grad_dot_feats = at::bmm(grad_feats_output.reshape({num_voxels, 1, feats_dim}), grid_feats.reshape({num_voxels, feats_dim, 1})).view_as(density);

            dim3 block(warp_size, warps_per_block, 1);
            ComputeGaussian2VoxelGradientTileWithFeats<float><<<num_gaussians, block, shared_bytes, s.stream()>>>(
                grad_density_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                grad_feats_output.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
                means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                cov_inv.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                opacity.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                density.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                d_features.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_dot_feats.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                valid_voxel_lb.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                valid_voxel_ub.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                g2t_offsets.data_ptr<int>(),
                g2t_tile_ids.data_ptr<int>(),
                max_tiles_y,
                max_tiles_z,
                d_means_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                d_cov_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                d_opacity_grad.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                d_grid_feats_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
        }
        else{
            ComputeGaussian2VoxelGradientTile<float><<<num_gaussians, num_threads_per_block_backward, 0, s.stream()>>>(
                grad_density_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                means3D.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                cov_inv.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                opacity.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                voxel_range.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                voxel_size.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                valid_voxel_lb.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                valid_voxel_ub.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                g2t_offsets.data_ptr<int>(),
                g2t_tile_ids.data_ptr<int>(),
                max_tiles_y,
                max_tiles_z,
                d_means_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                d_cov_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                d_opacity_grad.packed_accessor32<float,1,torch::RestrictPtrTraits>()
            );
        }

        #ifdef CUDA_TIMER_ENABLED
        cudaEventRecord(cuda_stop, s.stream());
        cudaEventSynchronize(cuda_stop);
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, cuda_start, cuda_stop);

        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
        // std::cout << "Backward GPU Time: " << gpu_time << " ms" << std::endl;
        return std::make_tuple(d_means_grad, d_cov_grad, d_opacity_grad, d_grid_feats_grad, gpu_time / 1000.0f);
        #else
        return std::make_tuple(d_means_grad, d_cov_grad, d_opacity_grad, d_grid_feats_grad, 0.0f);
        #endif

        // Return the result
    }
}
}