

#include <torch/extension.h>
#include "cuda_gaussian_voxelizer.cuh"
#include "cuda_feats_loss.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_voxelizer_forward", &cudaprocess::gaussian_voxelizer::cuda_gaussian_voxelizer_forward,
        py::arg("means3D"),
        py::arg("opacity"),
        py::arg("covariances"),
        py::arg("voxel_size"),
        py::arg("voxel_range"),
        py::arg("features") = py::none(),
        R"pbdoc(
        Perform Gaussian voxelization forward pass.

        Args:
            means3D (Tensor): (N, 3) Gaussian means.
            opacity (Tensor): (N,) or (N,1) opacity values.
            covariances (Tensor): (N, 3, 3) covariance matrices.
            voxel_size (Tensor): (3,) voxel resolution.
            voxel_range (Tensor): (6,) [x_min, y_min, z_min, x_max, y_max, z_max].

        Returns:
            Tuple: density (X, Y, Z), cov_inv, valid_voxel_lb, valid_voxel_ub, num_voxels_per_gaussian, grid_feats (X, Y, Z, C),
                   g2t_offsets (N+1,) and g2t_tile_ids (Gaussian->Tile CSR for the backward), forward time.
        )pbdoc");
    
    m.def("gaussian_voxelizer_backward", &cudaprocess::gaussian_voxelizer::cuda_gaussian_voxelizer_backward,
        py::arg("grad_output"),
        py::arg("grad_feats_output"),
        py::arg("means3D"),
        py::arg("opacity"),
        py::arg("cov_inv"),
        py::arg("voxel_size"),
        py::arg("voxel_range"),
        py::arg("valid_voxel_lb"),
        py::arg("valid_voxel_ub"),
        py::arg("num_voxels_per_gaussian"),
        py::arg("g2t_offsets"),
        py::arg("g2t_tile_ids"),
        py::arg("density"),
        py::arg("grid_feats"),
        py::arg("features") = py::none(),
        R"pbdoc(
        Perform Gaussian voxelization backward pass (one Gaussian per block, based on the Gaussian->Tile CSR).

        Args:
            grad_output (Tensor): (num_voxels,) gradient from the next layer.
            means3D (Tensor): (N, 3) Gaussian means.
            opacity (Tensor): (N,) or (N,1) opacity values.
            cov_inv (Tensor): (N, 3, 3) inverse covariance matrices.
            voxel_size (Tensor): (3,) voxel resolution.
            voxel_range (Tensor): (6,) [x_min, y_min, z_min, x_max, y_max, z_max].
            valid_voxel_lb (Tensor): Lower bounds of valid voxels.
            valid_voxel_ub (Tensor): Upper bounds of valid voxels.
            num_voxels_per_gaussian (Tensor): Number of voxels per Gaussian.
            g2t_offsets (Tensor): (N+1,) index pointers of the Gaussian->Tile CSR from the forward.
            g2t_tile_ids (Tensor): tile ids of the Gaussian->Tile CSR from the forward.

        Returns:
            Tuple[Tensor]: Gradients for means3D, opacity, and covariances.
        )pbdoc");
    // m.def("local_aggregate_backward", &LocalAggregateBackwardCUDA);

    m.def("CudaFeatsCosinelossForward", &cudaprocess::feats_loss::CudaFeatsCosinelossForward,
        py::arg("voxel_label"),
        py::arg("occ_gt_mask_camera"),
        py::arg("grid_feats"),
        py::arg("pseudo_dino_gt"),
        R"pbdoc(
        Compute voxel-wise cosine loss.

        Args:
            voxel_label (Tensor): (X, Y, Z) voxel labels.
            occ_gt_mask_camera (Tensor): (X, Y, Z) occupancy ground truth mask.
            grid_feats (Tensor): (X, Y, Z, C) grid features.
            pseudo_dino_gt (Tensor): (X, Y, Z, C) pseudo DINO ground truth features.

        Returns:
            float: The computed loss value.
        )pbdoc");

    m.def("CudaFeatsCosinelossBackward", &cudaprocess::feats_loss::CudaFeatsCosinelossBackward,
        py::arg("grad_output"),
        py::arg("voxel_label"),
        py::arg("occ_gt_mask_camera"),
        py::arg("grid_feats"),
        py::arg("pseudo_dino_gt"),
        py::arg("num_valid_voxel"),
        R"pbdoc(
        Compute gradients for voxel-wise cosine loss.
        Args:
            grad_output (Tensor): (1,) gradient from the next layer.
            voxel_label (Tensor): (X, Y, Z) voxel labels.
            occ_gt_mask_camera (Tensor): (X, Y, Z) occupancy ground truth mask.
            grid_feats (Tensor): (X, Y, Z, C) grid features.
            pseudo_dino_gt (Tensor): (X, Y, Z, C) pseudo DINO ground truth features.
            num_valid_voxel (int): Number of valid voxels.
        Returns:
            Tuple[Tensor, float]: Gradient for grid features and a placeholder float.
        )pbdoc");
}