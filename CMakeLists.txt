cmake_minimum_required(VERSION 3.18)
project(cuda_gaussian_voxelizer LANGUAGES CXX CUDA)

# ======== C++ Standard ========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ======== Build Type ========
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# ======== CUDA Flags ========
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -DTORCH_USE_CUDA_DSA=1")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0 -DTORCH_USE_CUDA_DSA=1")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# ======== Optional NVTX ========
option(USE_NVTX "Enable NVTX markers for profiling (if available)" OFF)

if(USE_NVTX)
    find_library(CUDA_NVTX_LIBRARY
        NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
        PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
        PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
        DOC "Location of the CUDA Toolkit Extension (NVTX) library"
        NO_DEFAULT_PATH
    )

    if(CUDA_NVTX_LIBRARY)
        message(STATUS "NVTX found and will be used.")
        add_definitions(-DUSE_NVTX)
        set(USE_NVTX_AVAILABLE TRUE)
    else()
        message(WARNING "NVTX requested but not found. Continuing without NVTX.")
        set(USE_NVTX_AVAILABLE FALSE)
    endif()
endif()

# ======== Find Dependencies ========
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# ======== Include Directories ========
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/thirdparty/nvtx/c/include  # optional local nvtx headers
)

# ======== Source Files ========
set(SOURCES
    src/bindings.cpp
    src/cuda_gaussian_voxelizer.cu
    src/cuda_feats_loss.cu
)

# ======== Shared Library ========
add_library(cuda_gaussian_voxelizer SHARED ${SOURCES})
set_target_properties(cuda_gaussian_voxelizer PROPERTIES CUDA_ARCHITECTURES "86")

# ======== Link Libraries ========
target_link_libraries(cuda_gaussian_voxelizer
    PRIVATE
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
    pybind11::module
)

# Conditionally link NVTX
if(USE_NVTX AND USE_NVTX_AVAILABLE)
    target_link_libraries(cuda_gaussian_voxelizer PRIVATE ${CUDA_NVTX_LIBRARY})
endif()

# PyTorch compile definitions
target_compile_definitions(cuda_gaussian_voxelizer PRIVATE ${TORCH_CXX_FLAGS})

# ======== Test Executable ========
add_executable(test_feat test/test_feat.cpp)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib)

target_link_libraries(test_feat
    PRIVATE
    cuda_gaussian_voxelizer
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    pybind11::module
)

if(USE_NVTX AND USE_NVTX_AVAILABLE)
    target_link_libraries(test_feat PRIVATE ${CUDA_NVTX_LIBRARY})
endif()

target_include_directories(test_feat PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(test_feat PRIVATE ${TORCH_CXX_FLAGS})

# ======== Print Summary ========
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "USE_NVTX: ${USE_NVTX} (available: ${USE_NVTX_AVAILABLE})")