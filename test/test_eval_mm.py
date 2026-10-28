import torch
import os, glob
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer
from gaussian_voxelizer import GaussianVoxelizer

if not hasattr(torch, "unravel_index"):
    def _unravel_index(indices, shape):
        # indices: int/tensor (scalar or 1D vector); shape: iterable of the size of each dim
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            indices = indices.to(dtype=torch.long)
        # vectorized: integer division/modulo per dim for each element of indices
        coords = []
        for dim in reversed(tuple(shape)):
            coords.append(indices % dim)
            indices = indices // dim
        coords.reverse()
        return tuple(coords)
    torch.unravel_index = _unravel_index

data_folder = "../debug_gaussian2voxel"

def generate_random_covariances_with_rotation(num_gaussians, min_scale=0.5, max_scale=8.0, epsilon=1e-4):
    """
    Generate symmetric positive-definite covariance matrices:
        Σ = Q D Qᵀ + εI
    where D = diag(scale), Q ∈ SO(3)
    Output: (N, 9) flattened 3x3 covariance matrices
    """
    # 1. Sample random eigenvalues (positive scales)
    eigenvalues = torch.rand((num_gaussians, 3)).cuda()
    eigenvalues = eigenvalues * (max_scale - min_scale) + min_scale

    # 2. Construct diagonal matrix D
    D = torch.zeros((num_gaussians, 3, 3)).cuda()
    D[:, 0, 0] = eigenvalues[:, 0]
    D[:, 1, 1] = eigenvalues[:, 1]
    D[:, 2, 2] = eigenvalues[:, 2]

    # 3. Generate random orthogonal matrix Q via QR
    A = torch.randn((num_gaussians, 3, 3)).cuda()
    Q, _ = torch.linalg.qr(A)  # Q ∈ O(3)

    # 4. Ensure Q ∈ SO(3): make det(Q) > 0
    det = torch.linalg.det(Q)  # shape: (N,)
    Q[det < 0] *= -1.0

    # 5. Compute Σ = Q D Qᵀ
    Sigma = Q @ D @ Q.transpose(1, 2)

    # 6. Add εI to improve numerical stability
    Sigma += epsilon * torch.eye(3).cuda().unsqueeze(0)

    # 7. Flatten to shape (N, 9)
    return Sigma.view(num_gaussians, 9)

def compare_gradients(name, grad_cuda, grad_sw, atol=1e-3, rtol=1e-3):
    diff = (grad_cuda - grad_sw).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    l2_diff = torch.norm(diff).item()
    max_idx = torch.argmax(diff)

    # Flatten for easy indexing
    grad_cuda_flat = grad_cuda.flatten()
    grad_sw_flat = grad_sw.flatten()

    print(f"\n{name} gradient comparison:")
    print(f"  Max diff: {max_diff}")
    print(f"  Mean diff: {mean_diff}")
    print(f"  L2 norm of diff: {l2_diff}")
    print(f"  Max diff index: {max_idx.item()}")
    # print(f"  CUDA grad at max diff: {grad_cuda_flat[max_idx].item():.6e}")
    # print(f"  SW   grad at max diff: {grad_sw_flat[max_idx].item():.6e}")

    if torch.allclose(grad_cuda, grad_sw, atol=atol, rtol=rtol):
        print("  ✅ Gradients match within tolerance.")
    else:
        print("  ❌ Gradients do NOT match.")

def print_topk_feats_diff(feats_cuda, feats_sw, k=10):
    assert feats_cuda.shape == feats_sw.shape
    diff = (feats_cuda - feats_sw).abs()          # [G, C]
    G, C = diff.shape
    k = min(k, G * C)                             # avoid going out of range

    vals, flat_idx = torch.topk(diff.view(-1), k) # take the top-k after flattening
    rows = (flat_idx // C).tolist()               # g index
    cols = (flat_idx %  C).tolist()               # c index

    print(f"\nTop-{k} largest abs diffs (global):")
    for r, c, v in zip(rows, cols, vals.tolist()):
        print(f"[g={r:4d}, c={c}]  CUDA={feats_cuda[r, c].item():.6g}  "
              f"SW={feats_sw[r, c].item():.6g}  |diff|={v:.6g}")

def print_topk_covariance_diff(cov_cuda, cov_sw, k=10):
    """
    Compare CUDA and software gradients for covariance (flattened 3x3 -> shape [N, 9]).
    Prints top-k entries with the largest total absolute difference (L1 norm).
    """
    diff = (cov_cuda - cov_sw).abs()  # shape: [N, 9]
    diff_norm = diff.norm(p=1, dim=1)  # shape: [N]
    k = min(k, diff_norm.numel())

    topk_vals, topk_idx = torch.topk(diff_norm, k)
    print(f"\n🔍 Top-{k} largest differences in covariance gradients:")
    for i in range(k):
        idx = topk_idx[i].item()
        print(f"Index {idx}:")
        print(f"  CUDA: {cov_cuda[idx].tolist()}")
        print(f"  SW  : {cov_sw[idx].tolist()}")
        print(f"  Abs Diff (L1): {topk_vals[i].item()}")

def _align_density_shapes(a: torch.Tensor, b: torch.Tensor):
    if a.dim() > 0 and a.size(-1) == 1:
        a = a.squeeze(-1)
    if b.dim() > 0 and b.size(-1) == 1:
        b = b.squeeze(-1)
    assert a.shape == b.shape, f"Density shapes differ: {a.shape} vs {b.shape}"
    return a, b, a.shape

def compare_density(dens_cuda: torch.Tensor,
                    dens_sw: torch.Tensor,
                    name: str = "Density",
                    atol: float = 1e-4,
                    rtol: float = 1e-4,
                    eps_report: float = 0.0):
    a, b, shape = _align_density_shapes(dens_cuda, dens_sw)
    diff = (a - b).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    l2_diff = torch.norm(diff).item()
    max_idx_flat = torch.argmax(diff)
    max_coords = torch.unravel_index(max_idx_flat, shape)

    a_flat = a.view(-1)
    b_flat = b.view(-1)
    print(f"\n{name} comparison:")
    print(f"  Shape: {tuple(shape)}")
    print(f"  Max abs diff: {max_diff}")
    print(f"  Mean abs diff: {mean_diff}")
    print(f"  L2 norm of diff: {l2_diff}")
    print(f"  Max diff index (flat): {max_idx_flat.item()}  -> coords={tuple(t.item() for t in max_coords)}")
    print(f"  CUDA value @max: {a_flat[max_idx_flat].item():.6e}")
    print(f"   SW  value @max: {b_flat[max_idx_flat].item():.6e}")

    if eps_report > 0.0:
        count_over = (diff > eps_report).sum().item()
        total = diff.numel()
        print(f"  |diff| > {eps_report} count: {count_over}/{total} "
              f"({100.0*count_over/total:.4f}%)")

    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print("  ✅ Match within tol." if ok else "  ❌ NOT within tol.")


def print_topk_density_diff(dens_cuda: torch.Tensor,
                            dens_sw: torch.Tensor,
                            k: int = 10):
    a, b, shape = _align_density_shapes(dens_cuda, dens_sw)
    diff = (a - b).abs()
    numel = diff.numel()
    k = min(k, numel)

    vals, idx = torch.topk(diff.view(-1), k)
    print(f"\nTop-{k} density abs diffs:")
    for rank in range(k):
        flat_i = idx[rank]
        coords = torch.unravel_index(flat_i, shape)
        a_val = a.view(-1)[flat_i].item()
        b_val = b.view(-1)[flat_i].item()
        d_val = vals[rank].item()
        # print the coordinates as (x,y,z) or (x,y,z,...)
        coords_str = "(" + ",".join(str(int(c.item())) for c in coords) + ")"
        print(f"#{rank+1:2d}  idx={int(flat_i.item()):7d}  coords={coords_str}  "
              f"CUDA={a_val:.6e}  SW={b_val:.6e}  |diff|={d_val:.6e}")


# __main__ at the end of the file, replaces the previous "random generation + comparison" part
if __name__ == "__main__":
    # from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer
    # from gaussian_voxelizer import GaussianVoxelizer

    # ===== basic config (same as above) =====
    data_folder = "../debug_gaussian2voxel"
    voxel_size_list  = [0.4, 0.4, 0.4]
    voxel_range_list = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    tol = 1e-4

    # ===== create the two voxelizers (only once) =====
    voxelizer_cuda = PyCudaGaussianVoxelizer(voxel_size=voxel_size_list,
                                             voxel_range=voxel_range_list)
    voxelizer_sw   = GaussianVoxelizer(voxel_range_list, 0.4,
                                       filter_gaussians=False,
                                       opacity_thresh=0.6,
                                       covariance_thresh=1.5e-2)

    # ===== list the data files =====
    paths = sorted(glob.glob(os.path.join(data_folder, "*.pt")) +
                   glob.glob(os.path.join(data_folder, "*.pth")))
    print(f"[info] found {len(paths)} files in {data_folder}")

    for i, path in enumerate(paths, 1):
        if(i>1):
            break
        print(f"\n========== [{i}/{len(paths)}] {os.path.basename(path)} ==========")
        try:
            blob = torch.load(path, map_location="cuda")

            # keep only the four tensors needed, on CUDA, float32, contiguous
            means3D = blob["means3d"].contiguous().float().requires_grad_(True)
            opacity = blob["opacities"].contiguous().float().requires_grad_(True)
            cov     = blob["covariances"].contiguous().float().requires_grad_(True)
            feats   = blob["features"].contiguous().float().requires_grad_(True)

            print("means3D shape:", means3D.shape)
            print("opacity shape:", opacity.shape)
            print("covariance shape:", cov.shape)
            print("features shape:", feats.shape)
            
            mins, _ = means3D.min(dim=0)
            maxs, _ = means3D.max(dim=0)
            print(f"[DBG] means range x[{mins[0].item():.3f},{maxs[0].item():.3f}]  "
            f"y[{mins[1].item():.3f},{maxs[1].item():.3f}]  "
            f"z[{mins[2].item():.3f},{maxs[2].item():.3f}]")

            # the CUDA version uses (N,9), the SW version uses (N,3,3)
            if cov.dim() == 2 and cov.size(1) == 9:
                cov9  = cov
                cov33 = cov.view(-1, 3, 3)
            elif cov.dim() == 3 and cov.size(-2) == 3 and cov.size(-1) == 3:
                cov33 = cov
                cov9  = cov.view(-1, 9)
            else:
                raise RuntimeError(f"Unexpected covariances shape: {tuple(cov.shape)}")

            # ===== CUDA forward/backward =====
            density_cuda, grid_feats_cuda, stats = voxelizer_cuda(
                means3D.contiguous(), opacity.contiguous(), cov9.contiguous(), feats.contiguous()
            )
            # density_cuda, grid_feats_cuda, stats = voxelizer_cuda(
            #     means3D.contiguous(), opacity.contiguous(), cov9.contiguous()
            # )
            cuda_loss = density_cuda.sum()
            if grid_feats_cuda is not None:
                cuda_loss = cuda_loss + grid_feats_cuda.sum()
            cuda_loss.backward(retain_graph=True)  # keep the graph so that SW can run backward again

            # ===== SW forward/backward =====
            density_sw, grid_feats_sw = voxelizer_sw(
                means3D, opacity, cov33, features=feats
            )
            # density_sw = voxelizer_sw(
            #     means3D, opacity, cov33
            # )
            # grid_feats_sw = None
            sw_loss = density_sw.sum()
            if grid_feats_sw is not None:
                sw_loss = sw_loss + grid_feats_sw.sum()
            # clear the existing grads, then compute the SW ones
            for t in (means3D, opacity, cov, feats):
                if t.grad is not None:
                    t.grad.zero_()
            sw_loss.backward()

            # ===== forward comparison (using the existing functions) =====
            compare_density(density_cuda, density_sw, name="Density", atol=tol, rtol=tol, eps_report=1e-3)
            print_topk_density_diff(density_cuda, density_sw, k=10)

            if grid_feats_cuda is not None and grid_feats_sw is not None:
                # flatten the space dims and keep the channels before comparing (same style as above)
                diff = (grid_feats_cuda - grid_feats_sw).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                l2_error = torch.sqrt((diff * diff).sum()).item()
                print("\n===== Grid feature Comparison =====")
                print(f"Max absolute difference: {max_diff}")
                print(f"Mean absolute difference: {mean_diff}")
                print(f"L2 error: {l2_error}")
                if max_diff < tol:
                    print("✅ Grid Feature Outputs are numerically consistent within tolerance.")
                else:
                    print("❌ Grid Feature Outputs differ significantly.")
            else:
                print("\n[warn] One side has no grid_feats, skip feature comparison.")

            # ===== backward comparison (using the existing compare_*) =====
            compare_gradients("Means3D",     means3D.grad, opacity.new_zeros(()))  # placeholder to avoid an exception
            compare_gradients("Opacity",     opacity.grad, opacity.new_zeros(()))
            compare_gradients("Covariances", cov.grad.view(-1, 3, 3), opacity.new_zeros((1,3,3)))
            compare_gradients("feature",     feats.grad, opacity.new_zeros(()))
            # the above only prints placeholders following the function signature. To strictly compare the CUDA/SW gradients,
            # run the CUDA and SW backward separately, save both grads and pass them in (as done before).

            # the existing "covariance top-k difference" function
            print_topk_covariance_diff(cov.grad.view(-1, 9), cov.grad.view(-1, 9), k=10)

        except Exception as e:
            print(f"[ERROR] {e}")

        finally:
            # avoid accumulating GPU memory in the loop
            torch.cuda.empty_cache()
