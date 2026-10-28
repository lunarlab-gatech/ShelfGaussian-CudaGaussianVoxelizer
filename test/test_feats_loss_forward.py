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
    
    
if __name__ == "__main__":
    # torch.manual_seed(42)
    device = "cuda"

    # ----- Grid spec -----
    voxel_size  = [0.4, 0.4, 0.4]
    voxel_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    X = int(round((voxel_range[3] - voxel_range[0]) / voxel_size[0]))  # 200
    Y = int(round((voxel_range[4] - voxel_range[1]) / voxel_size[1]))  # 200
    Z = int(round((voxel_range[5] - voxel_range[2]) / voxel_size[2]))  # 16
    C = 512  # keep moderate to avoid OOM; raise if you have VRAM

    # ----- Random predicted features -----
    grid_feats = torch.randn(X, Y, Z, C, device=device, dtype=torch.float32)

    # ----- Build masks -----
    p = 0.25
    voxels_label_full   = (torch.rand(X, Y, Z, device=device) < p)        # [X,Y,Z] bool
    occ_gt_mask_camera  = (torch.rand(1, X, Y, Z, device=device) < 0.7)   # [1,X,Y,Z] bool
    print("Valid voxels:", int((voxels_label_full & occ_gt_mask_camera.squeeze(0)).sum().item()))

    # ----- CUDA voxelizer -----
    voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size, voxel_range=voxel_range).to(device)
    
    loss_calculator = PyCudaFeatsCosineLoss(voxel_size=voxel_size, voxel_range=voxel_range).to(device)

    def run_case(title, pseudo_tensor_no_batch):
        """
        pseudo_tensor_no_batch: [X,Y,Z,C] float32
        """
        # match your call-site: add a leading dim so we can .squeeze(0)
        pseudo_dino_gt = pseudo_tensor_no_batch.unsqueeze(0).contiguous()

        # --- PyTorch path (KEEP THESE TWO LINES EXACTLY) ---
        t0 = time.time()
        feat_mask = voxels_label_full * occ_gt_mask_camera.squeeze(0)
        loss_pt = feat_loss(grid_feats.to(torch.float16), pseudo_dino_gt.squeeze(0).to(torch.float16), feat_mask)
        t1 = time.time()
        pt_time = t1 - t0

        # --- CUDA path ---
        loss_cuda, t_cuda = loss_calculator(
            voxels_label_full.float(),            # [X,Y,Z] float
            occ_gt_mask_camera.squeeze(0),        # [X,Y,Z] bool
            grid_feats,                            # [X,Y,Z,C] float32
            pseudo_dino_gt.squeeze(0)              # [X,Y,Z,C] float32
        )

        # --- Report ---
        diff = (loss_pt - loss_cuda).abs().item()
        print(f"{title}:")
        print(f"  PyTorch loss = {float(loss_pt):.6f}   (time {pt_time:.4f}s)")
        print(f"  CUDA    loss = {float(loss_cuda):.6f} (forward time {t_cuda[0].item():.4f}s)")
        print(f"  |diff|       = {diff:.6e}")

    # ----- Case A: GT = prediction (loss ≈ 0) -----
    run_case("Case A (GT = pred)", grid_feats)

    # ----- Case B: GT = -prediction (loss ≈ 2) -----
    run_case("Case B (GT = -pred)", -grid_feats)

    # ----- Case C: GT = pred + noise (loss in (0,1)) -----
    noise_level = 0.1
    run_case("Case C (GT = pred + noise)", grid_feats + noise_level * torch.randn_like(grid_feats))