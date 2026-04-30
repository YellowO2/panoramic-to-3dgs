import torch
import numpy as np
import math
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform
from datatype import View
from components.SplatProcessor.utils import (
    bilinear_interpolate_grid,
    project_gaussians_to_2d,
    compute_per_point_scales,
    project_world_cloud_to_view,
    rotate_to_pose,
    scale_gaussians,
    measure_nearest_z,
    trim_by_fov,
    trim_by_max_depth,
    merge,
    apply_smooth_alignment,
)


class SplatProcessor:
    MAX_DEPTH = 15.0  # metres; increase to keep more background, decrease to cut sky/far objects

    def __init__(self, grid_resolution=8, detail_weight=0.0):
        self.grid_resolution = grid_resolution
        self.detail_weight = detail_weight

    def _build_scale_grid(
        self,
        px_ok: np.ndarray,
        py_ok: np.ndarray,
        raw_scale_ok: np.ndarray,
        grid_cells_x: int,
        grid_cells_y: int,
        image_width: int,
        image_height: int,
        median_scale: float,
    ):
        """Step 3: Distribute point scales into a grid and compute cell medians."""
        cell_width = image_width / grid_cells_x
        cell_height = image_height / grid_cells_y
        grid = np.full((grid_cells_y, grid_cells_x), median_scale, dtype=np.float32)

        for gy in range(grid_cells_y):
            for gx in range(grid_cells_x):
                in_cell = (
                    (px_ok >= gx * cell_width)
                    & (px_ok < (gx + 1) * cell_width)
                    & (py_ok >= gy * cell_height)
                    & (py_ok < (gy + 1) * cell_height)
                )
                if int(in_cell.sum()) >= 8:
                    cell_scales = raw_scale_ok[in_cell]
                    cl, ch = np.quantile(cell_scales, [0.1, 0.9])
                    cell_trimmed = cell_scales[
                        (cell_scales >= cl) & (cell_scales <= ch)
                    ]
                    if cell_trimmed.size > 0:
                        grid[gy, gx] = float(np.median(cell_trimmed))

        return (
            np.clip(grid, median_scale * 0.1, median_scale * 10.0),
            cell_width,
            cell_height,
        )

    # ====== I commented these alignment algo out cause they SUCKED!! =====
    # def align_gaussians_to_depth(
    #     self,
    #     gaussians: Gaussians3D,
    #     reference_depth: np.ndarray,
    #     focal_x_px: float,
    #     focal_y_px: float,
    #     image_width: int,
    #     image_height: int,
    # ) -> Gaussians3D:
    #     """Align Gaussian depths to a reference depth map using a smooth scale grid."""
    #     # 1. Projection (from utils)
    #     pixel_x, pixel_y, depth_z, radial, valid = project_gaussians_to_2d(
    #         gaussians, focal_x_px, focal_y_px, image_width, image_height
    #     )

    #     num_valid = int(valid.sum())
    #     if num_valid < 64:
    #         return gaussians

    #     # 2. Compute point scales (from utils)
    #     raw_scale_ok, median_scale, ok = compute_per_point_scales(
    #         pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=True
    #     )

    #     if raw_scale_ok is None:
    #         return gaussians

    #     # 3. Build grid (Local logic)
    #     grid_cells_x = max(1, int(self.grid_resolution))
    #     grid_cells_y = max(1, int(round(grid_cells_x * (image_height / max(1, image_width)))))

    #     grid, cell_width, cell_height = self._build_scale_grid(
    #         pixel_x[valid][ok], pixel_y[valid][ok], raw_scale_ok,
    #         grid_cells_x, grid_cells_y, image_width, image_height, median_scale
    #     )

    #     # 4. Apply smooth alignment (Local logic)
    #     return self._apply_smooth_alignment(
    #         gaussians, grid, pixel_x, pixel_y, cell_width, cell_height,
    #         grid_cells_x, grid_cells_y, image_width, image_height,
    #         raw_scale_ok, valid, ok, median_scale
    #     )

    # def align_gaussians_global_scale_da3(
    #     self,
    #     gaussians: Gaussians3D,
    #     reference_depth: np.ndarray,
    #     focal_x_px: float,
    #     focal_y_px: float,
    #     image_width: int,
    #     image_height: int,
    # ) -> Gaussians3D:
    #     """Scale all Gaussians by a single global median scale derived from the reference depth."""
    #     pixel_x, pixel_y, depth_z, radial, valid = project_gaussians_to_2d(
    #         gaussians, focal_x_px, focal_y_px, image_width, image_height
    #     )
    #     if int(valid.sum()) < 64:
    #         return gaussians

    #     # use_radial=False because reference_depth stores Z (camera-plane depth)
    #     raw_scale_ok, median_scale, ok = compute_per_point_scales(
    #         pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=False
    #     )
    #     if raw_scale_ok is None or median_scale <= 0:
    #         return gaussians
    #     print(f"Global scale factor: {median_scale:.3f}")
    #     return scale_gaussians(gaussians, median_scale)
    # ==== End of alignment algo that sucked ===

    def align_gaussians_per_point(
        self,
        gaussians: Gaussians3D,
        reference_depth: np.ndarray,
        focal_x_px: float,
        focal_y_px: float,
        image_width: int,
        image_height: int,
    ) -> Gaussians3D:
        """Scale each Gaussian by its own DA3 depth ratio; median scale for unmatched points."""
        pixel_x, pixel_y, depth_z, radial, valid = project_gaussians_to_2d(
            gaussians, focal_x_px, focal_y_px, image_width, image_height
        )
        if int(valid.sum()) < 64:
            return gaussians

        raw_scale_ok, median_scale, ok = compute_per_point_scales(
            pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=False
        )
        if raw_scale_ok is None or median_scale <= 0:
            return gaussians

        per_point_scale = np.full(pixel_x.shape[0], median_scale, dtype=np.float32)
        valid_indices = np.where(valid)[0]
        ok_indices = np.where(ok)[0]
        per_point_scale[valid_indices[ok_indices]] = raw_scale_ok

        device = gaussians.mean_vectors.device
        scale_tensor = torch.tensor(per_point_scale, dtype=torch.float32, device=device).unsqueeze(1)
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors * scale_tensor,
            singular_values=gaussians.singular_values * scale_tensor,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )

    def align_gaussians_voronoi(
        self,
        gaussians: Gaussians3D,
        reference_depth: np.ndarray,
        focal_x_px: float,
        focal_y_px: float,
        image_width: int,
        image_height: int,
    ) -> Gaussians3D:
        """
        Per-Gaussian DA3 alignment via Voronoi cells.
        Pre-computes a label map (distance transform) so each pixel instantly knows
        its nearest DA3 anchor. Within each cell all Gaussians share the median scale
        so relative structure is preserved. Gaussians beyond diagonal/4 get global median.
        """
        # --- Step 1: extract DA3 anchor pixels from the sparse depth map ---
        da3_v_full, da3_u_full = np.where(reference_depth > 0)
        n_anchors = len(da3_u_full)
        print(f"  [Voronoi] DA3 anchors in view: {n_anchors}")
        if n_anchors < 4:
            print(f"  [Voronoi] Too few anchors, skipping alignment.")
            return gaussians

        # --- Step 2: project Gaussians to 2D ---
        pixel_x, pixel_y, depth_z, _, valid = project_gaussians_to_2d(
            gaussians, focal_x_px, focal_y_px, image_width, image_height
        )
        n_valid = int(valid.sum())
        print(f"  [Voronoi] Valid Gaussians: {n_valid} / {len(pixel_x)}")
        if n_valid < 16:
            return gaussians

        # --- Step 3: choose grid resolution from anchor density and aspect ratio ---
        aspect = image_width / image_height
        grid_w = int(min(math.sqrt(n_anchors * aspect), image_width))
        grid_h = int(min(math.sqrt(n_anchors / aspect), image_height))
        grid_w, grid_h = max(grid_w, 4), max(grid_h, 4)
        print(f"  [Voronoi] Grid: {grid_w}x{grid_h} (view: {image_width}x{image_height})")

        # scale anchor coords to grid space
        sx, sy = grid_w / image_width, grid_h / image_height
        da3_u_g = np.clip((da3_u_full * sx).astype(np.int32), 0, grid_w - 1)
        da3_v_g = np.clip((da3_v_full * sy).astype(np.int32), 0, grid_h - 1)
        # rebuild reference depth on the grid (min-z wins per cell)
        grid_depth = np.zeros((grid_h, grid_w), dtype=np.float32)
        np.maximum.at(grid_depth, (da3_v_g, da3_u_g), reference_depth[da3_v_full, da3_u_full])

        # --- Step 4: pre-compute Voronoi label map via distance transform on the small grid ---
        anchor_mask = grid_depth == 0
        dist_map, nearest = distance_transform_edt(anchor_mask, return_indices=True)
        max_dist_grid = math.sqrt(grid_w ** 2 + grid_h ** 2) / 4.0

        # --- Step 5: look up anchor ref-depth for each valid Gaussian ---
        valid_idx = np.where(valid)[0]
        valid_px_g = np.clip((pixel_x[valid_idx] * sx).astype(np.int32), 0, grid_w - 1)
        valid_py_g = np.clip((pixel_y[valid_idx] * sy).astype(np.int32), 0, grid_h - 1)
        valid_dz = np.clip(depth_z[valid_idx], 1e-6, None)

        anchor_row = nearest[0, valid_py_g, valid_px_g]
        anchor_col = nearest[1, valid_py_g, valid_px_g]
        ref_z = grid_depth[anchor_row, anchor_col].astype(np.float32)
        raw_scales = ref_z / valid_dz

        # --- Step 6: cell-wise median (vectorised via sorting) ---
        anchor_flat = anchor_row * grid_w + anchor_col
        sort_idx = np.argsort(anchor_flat)
        sorted_anchors = anchor_flat[sort_idx]
        sorted_scales = raw_scales[sort_idx]
        boundaries = np.flatnonzero(np.diff(sorted_anchors)) + 1
        groups = np.split(sorted_scales, boundaries)
        unique_anchors = sorted_anchors[np.concatenate([[0], boundaries])]
        cell_median_vals = np.array([np.median(g) for g in groups], dtype=np.float32)

        global_median = float(np.median(cell_median_vals))
        print(f"  [Voronoi] Cells: {len(unique_anchors)}, global median scale: {global_median:.4f}")
        if global_median <= 0:
            return gaussians

        anchor_to_median = dict(zip(unique_anchors, cell_median_vals))

        # --- Step 7: build per-Gaussian scale array ---
        per_gauss_scale = np.full(len(pixel_x), global_median, dtype=np.float32)
        gauss_dists = dist_map[valid_py_g, valid_px_g]
        within = gauss_dists <= max_dist_grid
        within_idx = valid_idx[within]
        within_anchors = anchor_flat[within]
        per_gauss_scale[within_idx] = np.array(
            [anchor_to_median[a] for a in within_anchors], dtype=np.float32
        )

        # --- Step 7: apply ---
        device = gaussians.mean_vectors.device
        scale_tensor = torch.tensor(per_gauss_scale, dtype=torch.float32, device=device).unsqueeze(1)
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors * scale_tensor,
            singular_values=gaussians.singular_values * scale_tensor,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )

    def align_splats_by_near_edge(
        self, views: list[View], splats_list: list[Gaussians3D]
    ) -> list[Gaussians3D]:
        """
        Scale each splat so all nearest-Z distances match the median across slices.
        Uses 0.1th percentile Z (≈ ground distance) as the scale proxy.
        Assumes trim_by_max_depth has already been applied (camera space, before pose).
        """
        nearest_zs = []
        for view, splat in zip(views, splats_list):
            z = measure_nearest_z(splat)
            nearest_zs.append(z)
            print(
                f"  Nearest Z [{view.yaw:+.0f}°]: {f'{z:.3f}' if z is not None else 'N/A'}"
            )

        valid_zs = [z for z in nearest_zs if z is not None]
        if not valid_zs:
            print("Near edge alignment: no valid measurements, skipping.")
            return splats_list

        target = float(np.median(valid_zs))
        print(f"Near edge target Z: {target:.3f}")

        result = []
        for splat, z in zip(splats_list, nearest_zs):
            if z is not None and z > 1e-6:
                result.append(scale_gaussians(splat, target / z))
            else:
                result.append(splat)
        return result

    def process(
        self,
        views: list[View],
        splats_list: list[Gaussians3D],
        pano_poses: dict = None,
        da3_world_pts: np.ndarray = None,
        scale_mode: str = "da3",
    ) -> Gaussians3D:
        """Main processing loop: align, trim, pose, and merge.
        scale_mode: 'da3' uses DA3 world cloud projection; 'bottom_edge' uses bottom-edge width matching.
        """
        processed_splats = []

        # Pre-compute per-view poses (needed for da3 mode inside loop)
        view_poses = []
        for view in views:
            R_local = Rotation.from_euler(
                "yx", [view.yaw, view.pitch], degrees=True
            ).as_matrix()
            pano_data = pano_poses.get(view.pano_id) if pano_poses else None
            center = pano_data["center"] if pano_data else None
            pano_rot = pano_data["rotation"] if pano_data else None
            R_c2w = pano_rot.T @ R_local if pano_rot is not None else R_local
            view_poses.append((R_local, center, pano_rot, R_c2w))

        # Step 1: trim far points first (camera space)
        trimmed = [trim_by_max_depth(splat, self.MAX_DEPTH) for splat in splats_list]

        # Step 2: scale alignment
        if scale_mode == "near_edge":
            print("--- Scale mode: near edge width ---")
            trimmed = self.align_splats_by_near_edge(views, trimmed)
        else:
            print("--- Scale mode: DA3 projection ---")
            for i, (view, splat) in enumerate(zip(views, trimmed)):
                _, center, _, R_c2w = view_poses[i]
                ref_depth = None
                pano_pts = da3_world_pts.get(view.pano_id) if isinstance(da3_world_pts, dict) else da3_world_pts
                if pano_pts is not None and center is not None:
                    ref_depth = project_world_cloud_to_view(
                        pano_pts, center, R_c2w, view
                    )
                elif view.depth is not None:
                    ref_depth = view.depth
                if ref_depth is not None:
                    trimmed[i] = self.align_gaussians_voronoi(
                        splat,
                        ref_depth,
                        view.focal_px,
                        view.focal_px,
                        int(view.width),
                        int(view.height),
                    )

        # Step 3: trim FOV edges, then apply pose
        for i, (view, splat) in enumerate(zip(views, trimmed)):
            _, center, _, R_c2w = view_poses[i]
            splat = trim_by_fov(splat, hfov_limit=view.hfov - 6.0)

            if center is not None:
                c2w = np.eye(4)
                c2w[:3, :3] = R_c2w
                c2w[:3, 3] = center
                splat = apply_transform(
                    splat,
                    torch.tensor(
                        c2w[:3, :],
                        dtype=torch.float32,
                        device=splat.mean_vectors.device,
                    ),
                )
            else:
                splat = rotate_to_pose(splat, yaw=view.yaw, pitch=view.pitch)

            processed_splats.append((view.pano_id, splat))

        per_pano_splats = {}
        for pano_id, splat in processed_splats:
            per_pano_splats.setdefault(pano_id, []).append(splat)
        per_pano_merged = {pid: merge(splats) for pid, splats in per_pano_splats.items()}

        return merge([s for _, s in processed_splats]), per_pano_merged
