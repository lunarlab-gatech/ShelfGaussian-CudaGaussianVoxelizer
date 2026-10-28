import torch
import time
import torch.nn.functional as F
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer, PyCudaFeatsCosineLoss

def feat_loss(voxel_feats_pred, voxel_feats_gt, mask):
    # Flatten voxels -> (N_voxels, C), Mask -> (N_voxels,)
    B, C = voxel_feats_pred.shape[:-1], voxel_feats_pred.shape[-1]
    flat_pred = voxel_feats_pred.reshape(-1, C)   # (N_voxels, 768)
    flat_gt   = voxel_feats_gt.reshape(-1, C)     # (N_voxels, 768)
    flat_mask = mask.reshape(-1)                  # (N_voxels,)

    # Index once (much faster than 4D boolean gather)
    masked_pred = flat_pred[flat_mask]   # (N_masked, 768)
    masked_gt   = flat_gt[flat_mask]     # (N_masked, 768)

    # Cast to fp16
    masked_pred = masked_pred.to(torch.float16)
    masked_gt   = masked_gt.to(torch.float16)

    # Normalize along feature dimension
    pred_norm = F.normalize(masked_pred, dim=-1, eps=1e-6)
    gt_norm   = F.normalize(masked_gt, dim=-1, eps=1e-6)

    # Cosine similarity
    cos_sim = torch.sum(pred_norm * gt_norm, dim=-1)  # (N_masked,)
    # print("cos_sim.shape", cos_sim.shape)
    return torch.mean(1 - cos_sim)

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

def run_case(title, grid_feats_base, pseudo_tensor_no_batch, voxelizer, loss_calculator, eps=1e-6):
    """
    grid_feats_base:         [X,Y,Z,C] float32 (baseline content for init)
    pseudo_tensor_no_batch:  [X,Y,Z,C] float32
    """
    # ----- Build masks -----
    X, Y, Z, C = grid_feats_base.shape
    p = 0.25
    voxels_label_full   = (torch.rand(X, Y, Z, device=device) < p)        # [X,Y,Z] bool
    occ_gt_mask_camera  = (torch.rand(1, X, Y, Z, device=device) < 0.7)   # [1,X,Y,Z] bool
    feat_mask = voxels_label_full & occ_gt_mask_camera.squeeze(0)         # [X,Y,Z] bool
    num_valid = int(feat_mask.sum().item())

    # -----------------------------
    # PyTorch forward + backward
    # -----------------------------
    grid_feats_pt = grid_feats_base.clone().detach().requires_grad_(True)

    t0 = time.time()
    loss_pt = feat_loss(grid_feats_pt, pseudo_tensor_no_batch, feat_mask)
    torch.cuda.synchronize()
    t1 = time.time()

    loss_pt.backward()
    torch.cuda.synchronize()
    t2 = time.time()

    grad_pt = grid_feats_pt.grad.detach().clone()

    # -----------------------------
    # CUDA forward + backward
    # -----------------------------
    grid_feats_cuda = grid_feats_base.clone().detach().requires_grad_(True)
    pseudo_dino_gt  = pseudo_tensor_no_batch.clone().detach()

    # forward
    t3 = time.time()
    out = loss_calculator(
        voxels_label_full.float(),            # [X,Y,Z] float
        occ_gt_mask_camera.squeeze(0),        # [X,Y,Z] bool
        grid_feats_cuda,                      # [X,Y,Z,C] float32
        pseudo_dino_gt                        # [X,Y,Z,C] float32
    )
    if isinstance(out, (tuple, list)):
        loss_cuda, stats = out[0], out[1]
        fwd_sec = float(stats[0].item()) if stats.numel() >= 1 else float('nan')
        bwd_sec = float(stats[1].item()) if stats.numel() >= 2 else float('nan')
    else:
        loss_cuda, fwd_sec, bwd_sec = out, float('nan'), float('nan')

    torch.cuda.synchronize()
    t4 = time.time()

    # backward
    grid_feats_cuda.grad = None
    loss_cuda.backward()
    torch.cuda.synchronize()
    t5 = time.time()

    grad_cuda = grid_feats_cuda.grad.detach().clone()

    # -----------------------------
    # Compare forward
    # -----------------------------
    f_diff = float((loss_pt - loss_cuda).abs().item())
    f_rel  = float(f_diff / (loss_pt.abs().item() + 1e-12))

    # -----------------------------
    # Compare gradients
    # -----------------------------
    mask4 = feat_mask.unsqueeze(-1).expand_as(grad_pt)  # [X,Y,Z,C]
    grad_pt_masked   = torch.where(mask4, grad_pt,   torch.zeros_like(grad_pt))
    grad_cuda_masked = torch.where(mask4, grad_cuda, torch.zeros_like(grad_cuda))

    g_abs_diff = (grad_pt_masked - grad_cuda_masked).abs()
    g_max_abs  = float(g_abs_diff.max().item())

    g_rel_diff = g_abs_diff / (grad_pt_masked.abs() + 1e-8)
    g_max_rel  = float(g_rel_diff.max().item())

    nz_pt   = (grad_pt_masked.abs() > 0).sum().item()
    nz_cuda = (grad_cuda_masked.abs() > 0).sum().item()

    print(f"\n=== {title} ===")
    print(f"Valid voxels: {num_valid}")
    print(f"PyTorch  forward: loss={float(loss_pt):.6f}, time={t1-t0:.4f}s, backward time={t2-t1:.4f}s")
    if not (fwd_sec != fwd_sec):  # not NaN
        print(f"CUDA     forward: loss={float(loss_cuda):.6f}, forward={fwd_sec:.4f}s, backward={bwd_sec:.4f}s")
    else:
        print(f"CUDA     forward: loss={float(loss_cuda):.6f}, time={t4-t3:.4f}s, backward time={t5-t4:.4f}s")

    print(f"Forward |diff| = {f_diff:.6e} (rel {f_rel:.6e})")
    # print(f"Grad max abs diff = {g_max_abs:.6e}, max rel diff = {g_max_rel:.6e}")
    compare_gradients("Grid feats", grad_cuda_masked, grad_pt_masked, atol=1e-3, rtol=1e-3)
    # number of non-zeros in gradients
    print(f"Grad nnz (pt/cuda) = {nz_pt}/{nz_cuda}")

    atol, rtol = 5e-5, 5e-4
    if not (f_diff <= atol or f_diff <= rtol * float(abs(loss_pt))):
        print("WARNING: forward mismatch beyond tolerance.")
    if not (g_max_abs <= 1e-3 or g_max_rel <= 1e-2):
        print("WARNING: gradient mismatch beyond tolerance.")


if __name__ == "__main__":
    device = "cuda"
    # ----- Grid spec -----
    voxel_size  = [0.4, 0.4, 0.4]
    voxel_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    X = int(round((voxel_range[3] - voxel_range[0]) / voxel_size[0]))  # 200
    Y = int(round((voxel_range[4] - voxel_range[1]) / voxel_size[1]))  # 200
    Z = int(round((voxel_range[5] - voxel_range[2]) / voxel_size[2]))  # 16
    C = 64

    # ----- Random predicted features -----
    grid_feats = torch.randn(X, Y, Z, C, device=device, dtype=torch.float32)

    # ----- Modules -----
    voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size, voxel_range=voxel_range).to(device)
    loss_calculator = PyCudaFeatsCosineLoss(voxel_size=voxel_size, voxel_range=voxel_range).to(device)

    # ----- Cases -----
    # A: GT = pred  (loss ~ 0)
    run_case("Case A (GT = pred)", grid_feats, grid_feats, voxelizer, loss_calculator, eps=1e-6)

    # B: GT = -pred (loss ~ 2)
    run_case("Case B (GT = -pred)", grid_feats, -grid_feats, voxelizer, loss_calculator, eps=1e-6)

    # C: GT = pred + noise (loss in (0, 1))
    noise_level = 0.1
    run_case("Case C (GT = pred + noise)",
             grid_feats,
             grid_feats + noise_level * torch.randn_like(grid_feats),
             voxelizer, loss_calculator, eps=1e-6)