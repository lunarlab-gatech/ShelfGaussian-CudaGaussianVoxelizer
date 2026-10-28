#ifndef CUDA_PROCESS_UTIL_H
#define CUDA_PROCESS_UTIL_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <curand.h>

namespace cudaprocess {

    // CUDA API error checking
    #define CHECK_CUDA(err)                                                         \
    do {                                                                            \
        cudaError_t err_ = (err);                                                   \
        if (err_ != cudaSuccess) {                                                  \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);           \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

    // curand API error checking
    #define CURAND_CHECK(err)                                                       \
    do {                                                                            \
        curandStatus_t err_ = (err);                                                \
        if (err_ != CURAND_STATUS_SUCCESS) {                                        \
            printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);         \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

    // CUDSS
    #define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            exit(-2); \
        } \
    } while(0);

    constexpr int stream_cnt = 7;
    struct CudaUtil {
        cudaStream_t streams_[stream_cnt];
        cudaEvent_t events_[stream_cnt];
        // float *tmp_res;
        // int *tmp_invalid_flag;

        CudaUtil() {
            for (int i = 0; i < stream_cnt; ++i) {
                CHECK_CUDA(cudaStreamCreate(&streams_[i]));
                CHECK_CUDA(cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming));
            }
            // CHECK_CUDA(cudaMalloc(&tmp_res, sizeof(float) * CUDA_SOLVER_POP_SIZE * stream_cnt));
            // CHECK_CUDA(cudaMalloc(&tmp_invalid_flag, sizeof(int) * CUDA_SOLVER_POP_SIZE));
    }

    ~CudaUtil() {
        for (int i = 0; i < stream_cnt; ++i) {
            // CHECK_CUDA(cudaStreamDestroy(streams_[i]));
            // CHECK_CUDA(cudaEventDestroy(events_[i]));
            cudaError_t err = cudaStreamDestroy(streams_[i]);
            if (err != cudaSuccess) {
                // std::cerr << "Failed to destroy stream[" << i << "]: " << cudaGetErrorString(err) << std::endl;
                printf("Failed to destroy stream[%d]: %s\n", i, cudaGetErrorString(err));
            }

            err = cudaEventDestroy(events_[i]);
            if (err != cudaSuccess) {
                // std::cerr << "Failed to destroy event[" << i << "]: " << cudaGetErrorString(err) << std::endl;
                printf("Failed to destroy event[%d]: %s\n", i, cudaGetErrorString(err));
            }
        }
        // CHECK_CUDA(cudaFree(tmp_res));
        // CHECK_CUDA(cudaFree(tmp_invalid_flag));
    }
};

}

#endif