#ifndef CUDAPROCESS_GAUSSIAN_VOXELIZER_UTILS_CUH
#define CUDAPROCESS_GAUSSIAN_VOXELIZER_UTILS_CUH

// #define NVTX_ENABLED
#ifdef NVTX_ENABLED
#include <nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#define NVTX_MARK(name) nvtxMarkA(name)
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#define NVTX_MARK(name)
#endif

#define CHECK_TENSOR(tensor, name) \
        std::cout << name << ": shape=" << tensor.sizes() \
                << ", dtype=" << tensor.dtype() \
                << ", device=" << tensor.device() \
                << ", is_contiguous=" << tensor.is_contiguous() \
                << std::endl;

#define CUDA_TIMER_ENABLED

// #define DEBUG_MODE

namespace cudaprocess {
    // namespace gaussian_voxelizer {

        // constexpr int num_gaussians = 1800;                                                                                     // Number of Gaussians for whole scene
        // constexpr float voxel_range[6] = {                                                                                      // Voxel range in meters
        //     -40.0f, -40.0f, -1.0f,                                                                                              // x_min, y_min, z_min
        //     40.0f, 40.0f, 5.4f                                                                                                  // x_max, y_max, z_max
        // };
        // // constexpr float voxel_size = 0.4f;                                                                                      // Voxel size in meters

        // constexpr float voxel_size[3] = {0.4f, 0.4f, 0.4f};                                                                     // Voxel size in meters for x, y, z dimensions

        // constexpr float opacity_threshold = 0.6f;                                                                               // Opacity threshold for Gaussian filtering
        // constexpr float covariance_threshold = 1.5e-2f;                                                                         // Minimum weight for a Gaussian to be considered

        // constexpr int voxel_dim_x = static_cast<int>((voxel_range[3] - voxel_range[0]) / voxel_size[0]);                           // Number of voxels in x dimension   
        // constexpr int voxel_dim_y = static_cast<int>((voxel_range[4] - voxel_range[1]) / voxel_size[1]);                           // Number of voxels in y dimension             
        // constexpr int voxel_dim_z = static_cast<int>((voxel_range[5] - voxel_range[2]) / voxel_size[2]);                           // Number of voxels in z dimension

        // constexpr int num_voxels = voxel_dim_x * voxel_dim_y * voxel_dim_z;                                                     // Total number of voxels in the scene
        
        // constexpr bool record_host_memory = false;                                                                              // Flag to record host memory for debugging

        constexpr int TILE_X = 4;                                                                                             // Number of tiles in x dimension
        constexpr int TILE_Y = 4;                                                                                             // Number of tiles in y dimension
        constexpr int TILE_Z = 4;                                                                                             // Number of tiles in z dimension

        // constexpr int MAX_TILES_X = 200 / TILE_X;                                                                                  // Maximum number of tiles in x dimension
        // constexpr int MAX_TILES_Y = 200 / TILE_Y;                                                                                  // Maximum number of tiles in y dimension
        // constexpr int MAX_TILES_Z = 16 / TILE_Z;                                                                                   // Maximum number of tiles in z dimension

        constexpr int dim3d = 3;                                                                                                // Dimension of the voxel (3D: x, y, z)
        constexpr int covariance_dim = 9;                                                                                       // Covariance matrix is 3x3, flattened to 9 elements

        constexpr int voxel_range_len = 6;                                                                                      // Length of the voxel range array
        constexpr int voxel_size_len = 3;                                                                                       // Length of the voxel size array

        constexpr int max_voxels_per_gaussian = 1024;                                                                           // Maximum number of voxels per Gaussian

        constexpr uint enable_backward = 1;                                                                                     // Flag to enable backward pass

        constexpr int recorded_cov_dims = 6;                                                                                     // Number of recorded covariance dimensions (3D mean + 3D covariance)

        // for ComputeGaussian2Voxel
        constexpr int num_block_per_gaussian = 8;                                                                                  // Number of blocks per Gaussian for parallel processing
        constexpr int num_threads_per_block = 1024;                                                                                 // Number of threads per block, this must be a power of 2 like 256, 512, 1024

        constexpr int warp_size = 32;

        // for the backward based on the Gaussian->Tile CSR (one Gaussian per block)
        constexpr int num_threads_per_block_backward = 256;                                                                     // Must be a multiple of warp_size
        constexpr int num_warps_per_block_backward = 8;                                                                         // Max warps (one voxel per warp) when the features are given

        constexpr float eps = 1e-6f;                                                                                         // Small epsilon value to prevent division by zero
        // *********************** DEVICE ************************
        // __constant__ float d_voxel_range[voxel_range_len];
        // __constant__ float d_voxel_size[voxel_size_len];

        // record
        // __constant__ float d_gaussians_cov_inv[num_gaussians * recorded_cov_dims];

        // record number of voxels will be affected by each Gaussian
        // __constant__ int d_num_voxels_per_gaussian[num_gaussians];
        // __constant__ int3 d_valid_voxel_lower_boundary[num_gaussians];                                                            // Minimum boundary for each Gaussian that affects the voxelization
        // __constant__ int3 d_valid_voxel_upper_boundary[num_gaussians];                                                            // Maximum boundary for each Gaussian that affects the voxelization
    // } // namespace gaussian_voxelizer
}
#endif