#ifndef CUDA_GAUSSIAN_VOXELIZER_DATA_TYPE_H
#define CUDA_GAUSSIAN_VOXELIZER_DATA_TYPE_H

namespace cudaprocess {
// *********************** BASE DATA STRUCTURE ***********************
template <typename T, int size>
struct CudaVector {
    T data[size];
    int len{size};
};

namespace gaussian_voxelizer {

    struct CudaVoxel {
        float3 center{0.0f, 0.0f, 0.0f};                    // Center of the voxel
        float density{0.0f};                                // Density of the voxel (occupancy logits)
        float opacity{0.0f};                                // Opacity of the voxel (record the max opacity of all Gaussians in the voxel)
        uint semantics{0};                                   // Semantics of the voxel (record the max semantics of all Gaussians in the voxel)
    };

    struct CudaGaussian {
        float3 mean;                                        // Mean of the Gaussian
        float covariance[9];                                // Covariance matrix of the Gaussian (flattened)
        float opacity;                                      // Opacity of the Gaussian
        int semantics;                                      // Semantics of the Gaussian
    };

    struct CudaVoxelGaussianPair {
        int voxel_idx[1024];                                // Index of the voxel
        int voxel_count;                                    // Number of Gaussians in the voxel
        int gaussian_idx;                                   // Index of the Gaussian
        int num_gaussians;                                   // Number of Gaussians in the voxel
    };
}
}

#endif