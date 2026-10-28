# Tests and benchmarks

This folder holds the correctness tests, benchmark scripts and reference implementations for CudaGaussianVoxelizer.

## Setup

Install the package first. See [Installation](../README.md#installation) in the main README.

```bash
pip install --no-build-isolation -e .
cd test
```

Run the scripts from inside `test/`, because some of them use paths relative to it. Some scripts need more than PyTorch:

| Extra need | Used by |
| --- | --- |
| `pip install taichi` | `test_forward.py`, `test_backward.py`, `benckmark.py` (they import `taichi_voxelizer.py`) |
| `pip install matplotlib` | `benckmark_plot.py` |
| Recorded Gaussians in `../debug_gaussian2voxel/` | `test_feat.py`, `test_eval_mm.py`, `test_eval_mm.cpp` |

`debug_gaussian2voxel/` is not part of the repo (it is gitignored). Each `.pt` / `.pth` file in it is a dict saved with `torch.save`, with the keys `means3d`, `opacities`, `covariances` and `features`.

## Start here

```bash
python test_reference.py
```

[test_reference.py](test_reference.py) compares the CUDA voxelizer against a brute-force float64 PyTorch implementation. It checks density, grid features, and the gradients of `means3D`, `opacity`, `covariances` and `features`, including Gaussians centered outside the voxel range. It needs no extra data or packages, prints PASS/FAIL for each check, and exits with code 1 if any check fails.

## Correctness tests

| Script | What it compares | Needs |
| --- | --- | --- |
| [test_reference.py](test_reference.py) | Density, features and all gradients against a brute-force float64 reference | Nothing extra |
| [test_forward.py](test_forward.py) | Forward density (9k Gaussians, no features) against GaussTR's PyTorch voxelizer and Taichi. It also prints the forward time of each. | Taichi |
| [test_backward.py](test_backward.py) | Density gradients (10k Gaussians) against GaussTR's PyTorch voxelizer | Taichi |
| [test_feat.py](test_feat.py) | Density, features and all gradients against GaussTR's PyTorch voxelizer, on the first recorded file | `debug_gaussian2voxel/` |
| [test_eval_mm.py](test_eval_mm.py) | The same as `test_feat.py`, looped over every recorded file | `debug_gaussian2voxel/` |
| [test_feats_loss_forward.py](test_feats_loss_forward.py) | `PyCudaFeatsCosineLoss` forward against a PyTorch cosine loss, for three targets: equal to the prediction (loss 0), negated (loss 2), and with added noise | Nothing extra |
| [test_feats_loss_backward.py](test_feats_loss_backward.py) | `PyCudaFeatsCosineLoss` gradients against PyTorch autograd, for the same three targets | Nothing extra |

## Benchmark

| File | What it is |
| --- | --- |
| [benckmark.py](benckmark.py) | Times the CUDA forward and backward, the GaussTR-style PyTorch splatting and the Taichi forward over 100 random scenes of 18k Gaussians. It saves the timings to `fwd_*.npy` / `bwd_*.npy` in the current folder. |
| `fwd_cuda.npy`, `bwd_cuda.npy`, `fwd_sw.npy`, `bwd_sw.npy`, `fwd_taichi.npy` | Timings saved by an earlier `benckmark.py` run |
| [benckmark_plot.py](benckmark_plot.py) | Plots CUDA against GaussTR (forward, backward and total time per iteration). The timings are hard-coded from an earlier run; the script does not read the `.npy` files. |

The numbers reported in the paper (Table 13) are in the [main README](../README.md#performance).

## Reference implementations

| File | What it is |
| --- | --- |
| [gaussian_voxelizer.py](gaussian_voxelizer.py) | GaussTR's PyTorch voxelizer with its mmdet3d dependencies removed. Most comparison tests use it as the "SW" reference. |
| [taichi_voxelizer.py](taichi_voxelizer.py) | The Taichi voxelizer used as a baseline |

## Other files

| File | What it is |
| --- | --- |
| [demo.py](demo.py) | A short usage example of `PyCudaGaussianVoxelizer` and `PyCudaFeatsCosineLoss` |
| [test_feat_tr.py](test_feat_tr.py) | Gradient checks on the PyTorch reference splatting only. It does not call the CUDA kernels. |
| [test_data_from_mmdetection.py](test_data_from_mmdetection.py) | Re-voxelizes `../debug_gaussian_voxelizer_output.pth` and prints the difference to the densities stored in it. That file was recorded on 2025-08-12, before the voxel-center and out-of-range fixes, so large differences are expected. Use `test_reference.py` to check correctness instead. |
| `test_forward.cpp`, `test_backward.cpp`, `test_feat.cpp`, `test_forward_result.cpp`, `test_eval_mm.cpp` | C++ drivers that call the kernels without Python, which is handy for `compute-sanitizer` or Nsight. Only `test_feat.cpp` has a CMake target (`test_feat` in [CMakeLists.txt](../CMakeLists.txt)). |
