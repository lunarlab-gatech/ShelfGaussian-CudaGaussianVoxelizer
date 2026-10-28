import torch.nn as nn
import torch
import torch.nn.functional as F

# from ._C import CudaGaussianVoxelizer
from ._C import gaussian_voxelizer_forward, gaussian_voxelizer_backward
from ._C import CudaFeatsCosinelossForward, CudaFeatsCosinelossBackward


class _CudaGaussianVoxelizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, opacity, covariances, voxel_size, voxel_range, features):
        # means32 = means3D.to(torch.float32).contiguous()
        # op32    = opacity.to(torch.float32).contiguous()
        # cov32   = covariances.to(torch.float32).contiguous()
        # voxsz32 = voxel_size.to(torch.float32).contiguous()
        # voxrg32 = voxel_range.to(torch.float32).contiguous()
        # print("means32 content:", means32)
        # print("opacity content:", op32)
        # print("cov32 content:", cov32)
        # print("voxsz32 content:", voxsz32)
        # print("voxrg32 content:", voxrg32)
        # density, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel = gaussian_voxelizer_forward(means32, op32, cov32, voxsz32, voxrg32)
        # ctx.save_for_backward(means32, op32, voxsz32, voxrg32, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel)
        # return density
        
        # g2t_offsets / g2t_tile_ids: Gaussian->Tile CSR, reused by the backward (one Gaussian per block)
        density, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel, grid_feats, g2t_offsets, g2t_tile_ids, fwd_sec = gaussian_voxelizer_forward(means3D, opacity, covariances, voxel_size, voxel_range, features)
        stats = torch.zeros(2, dtype=torch.float32, device='cpu')
        stats[0] = fwd_sec

        ctx.save_for_backward(means3D, opacity, voxel_size, voxel_range, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel, g2t_offsets, g2t_tile_ids, density, grid_feats, features, stats)
        ctx.mark_non_differentiable(stats)
        return density, grid_feats, stats

    @staticmethod
    def backward(ctx, grad_density_output, grad_feats_output, grad_stats):
        # means3D, opacity, voxel_size, voxel_range, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel = ctx.saved_tensors
        (means32, op32, voxsz32, voxrg32, cov_inv, valid_voxel_lb, valid_voxel_ub, num_valid_voxel, g2t_offsets, g2t_tile_ids, density, grid_feats, features, stats) = ctx.saved_tensors
        # print("grad_output shape:", grad_output.shape)
        gout32 = grad_density_output.to(torch.float32).contiguous()
        grad_feats_output = grad_feats_output.to(torch.float32).contiguous()
        grad_means3D, grad_cov, grad_opacity, grad_feats, bwd_sec = gaussian_voxelizer_backward(gout32, grad_feats_output, means32, op32, cov_inv, voxsz32, voxrg32, valid_voxel_lb, valid_voxel_ub, num_valid_voxel, g2t_offsets, g2t_tile_ids, density, grid_feats, features)
        stats[1] = bwd_sec
        # raise NotImplementedError("Backward not implemented yet.")
        grad = (grad_means3D, grad_opacity, grad_cov, None, None, grad_feats if features is not None else None)
        return grad

class _CudaFeatsCosinelossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxel_label, occ_gt_mask_camera, d_grid_feats, pseudo_dino_gt):
        loss, num_valid_voxel, fwd_sec = CudaFeatsCosinelossForward(voxel_label, occ_gt_mask_camera, d_grid_feats, pseudo_dino_gt)
        stats = torch.zeros(2, dtype=torch.float32, device='cpu')
        stats[0] = fwd_sec
        ctx.save_for_backward(voxel_label, occ_gt_mask_camera, d_grid_feats, pseudo_dino_gt, stats)
        ctx.num_valid_voxel = num_valid_voxel
        return loss, stats

    @staticmethod
    def backward(ctx, grad_output, grad_stats):
        voxel_label, occ_gt_mask_camera, d_grid_feats, pseudo_dino_gt, stats = ctx.saved_tensors
        num_valid_voxel = ctx.num_valid_voxel
        gout32 = grad_output.to(torch.float32).contiguous()
        grad_predict, bwd_sec = CudaFeatsCosinelossBackward(gout32, voxel_label, occ_gt_mask_camera, d_grid_feats, pseudo_dino_gt, num_valid_voxel)
        stats[1] = bwd_sec
        grad = (None, None, grad_predict, None)
        return grad

class PyCudaGaussianVoxelizer(nn.Module):
    def __init__(self, voxel_size=[0.4, 0.4, 0.4], voxel_range=[-40, -40, -1, 40, 40, 5.4]):
        super().__init__()
        # list to tensor device
        # self.voxel_size = torch.tensor(voxel_size).cuda()
        # self.voxel_range = torch.tensor(voxel_range).cuda()
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
        self.voxel_range = torch.tensor(voxel_range, dtype=torch.float32, device='cuda')

            
    def forward(self, means3D, opacity, covariances, features=None):
        return _CudaGaussianVoxelizeFunction.apply(
            means3D, opacity, covariances, self.voxel_size, self.voxel_range, features
        )
        
class PyCudaFeatsCosineLoss(nn.Module):
    def __init__(self, voxel_size=[0.4, 0.4, 0.4], voxel_range=[-40, -40, -1, 40, 40, 5.4]):
        super().__init__()
        # list to tensor device
        # self.voxel_size = torch.tensor(voxel_size).cuda()
        # self.voxel_range = torch.tensor(voxel_range).cuda()
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
        self.voxel_range = torch.tensor(voxel_range, dtype=torch.float32, device='cuda')

    def forward(self, voxel_label, occ_gt_mask_camera, d_grid_feats, pseudo_dino_gt):
        return _CudaFeatsCosinelossFunction.apply(
            voxel_label, occ_gt_mask_camera, d_grid_feats, pseudo_dino_gt
        )