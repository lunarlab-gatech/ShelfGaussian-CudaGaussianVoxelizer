from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
setup(
    name="CudaGaussianVoxelizer",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="CudaGaussianVoxelizer._C",
            sources=[
                "src/bindings.cpp",
                "src/cuda_gaussian_voxelizer.cu",
                "src/cuda_feats_loss.cu",
            ],
            include_dirs=[os.path.abspath("include")],
            # libraries=["nvtx3"],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math']
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

