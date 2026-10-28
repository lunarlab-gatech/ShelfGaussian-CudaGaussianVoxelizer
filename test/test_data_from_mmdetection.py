import torch
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(curr_dir)

pth_path = os.path.join(parent_dir, "debug_gaussian_voxelizer_output.pth")

data = torch.load(pth_path)

means3d = data["means3d"]        # shape: (B, N, num_queries, 3)
opacities = data["opacities"]    # shape: (B, N, num_queries, 1)
covariances = data["covariances"]  # shape: (B, N, num_queries, 9)

density_from_record = data["grid_density"]  # shape: (B, N, num_queries, 1)
density_tr_from_record = data["grid_density_TR"]  # shape: (B, N, num_queries, 1)

print("Means3D shape:", means3d.shape)
print("Means3D type:", means3d.dtype)
print("Means3D device:", means3d.device)
print("Opacities shape:", opacities.shape)
print("Opacities type:", opacities.dtype)
print("Opacities device:", opacities.device)
print("Covariances shape:", covariances.shape)
print("Covariances type:", covariances.dtype)
print("Covariances device:", covariances.device)

print("Density from record shape:", density_from_record.shape)
print("Density from record type:", density_from_record.dtype)
print("Density from record device:", density_from_record.device)
print("Density TR from record shape:", density_tr_from_record.shape)
print("Density TR from record type:", density_tr_from_record.dtype)
print("Density TR from record device:", density_tr_from_record.device)

voxel_size_list = [0.4, 0.4, 0.4]
voxel_range_list = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]



print(data.keys())

voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size_list, voxel_range=voxel_range_list)

density, _, _ = voxelizer(means3d.flatten(0, 2).float(), opacities.flatten(0, 3).float(), covariances.flatten(0, 2).flatten(1, 2).float())

print("Density re-computed shape:", density.shape)
print("Density re-computed type:", density.dtype)
print("Density re-computed device:", density.device)

# print("Density re-computed", density)

print("Density TR recorded result:", density_tr_from_record.squeeze(0).squeeze(-1))

print("Difference between recorded density and re-computed density:")
diff = (density_from_record - density).abs()
print("Max abs diff:", diff.max().item())
print("Mean abs diff:", diff.mean().item())


print("Difference between TR density and re-computed density:")
diff2 = (density_tr_from_record.squeeze(0).squeeze(-1) - density).abs()
print("Max abs diff TR:", diff2.max().item())
print("Mean abs diff TR:", diff2.mean().item())










######################################## BACKWARD TEST ########################################
loss = density.sum()
loss.backward()

print("check means3d grad:", means3d.grad)