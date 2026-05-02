import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform
from datatype import View
from components.SplatProcessor.utils import (
    project_world_cloud_to_view,
    rotate_to_pose,
    trim_by_fov,
    trim_by_max_depth,
    trim_by_pano_voronoi,
    correct_interpano_seams,
    merge,
)
from components.SplatProcessor.alignment import (
    align_near_edge,
    align_da3_per_point,
    align_da3_zslab,
    align_da3_2dgrid,
    align_da3_y_ground,
    elevation_estimate,
)


class SplatProcessor:
    MAX_DEPTH = 10.0

    def __init__(
        self,
        num_z_slabs: int = 500,
        num_fov_slabs: int = 50,
        smooth_sigma_m: float = 0.5,
        smooth_sigma_fov: float = 0.15,
        voronoi_buffer_m: float = 1.5,
    ):
        self.num_z_slabs = num_z_slabs
        self.num_fov_slabs = num_fov_slabs
        self.smooth_sigma_m = smooth_sigma_m
        self.smooth_sigma_fov = smooth_sigma_fov
        self.voronoi_buffer_m = voronoi_buffer_m

    def _resolve_ref_depth(
        self,
        view: View,
        center,
        R_c2w,
        scale_mode: str,
        da3_world_pts,
        all_da3_pts,
    ):
        """Return the reference depth map for one view, or None if unavailable."""
        if scale_mode in ("da3_zslab_global", "da3_2dgrid_global"):
            pts = all_da3_pts
        elif isinstance(da3_world_pts, dict):
            pts = da3_world_pts.get(view.pano_id)
        else:
            pts = da3_world_pts

        if pts is not None and center is not None:
            return project_world_cloud_to_view(
                pts, center, R_c2w, view, max_depth=self.MAX_DEPTH * 1.5
            )
        if view.depth is not None:
            return view.depth
        return None

    def _align_splat(
        self, splat: Gaussians3D, ref_depth, view: View, scale_mode: str
    ) -> Gaussians3D:
        """Dispatch to the correct alignment function for the given scale_mode."""
        fx = fy = view.focal_px
        w, h = int(view.width), int(view.height)
        match scale_mode:
            case "da3_per_point":
                return align_da3_per_point(
                    splat, ref_depth, fx, fy, w, h, smooth_sigma=self.smooth_sigma_fov
                )
            case "da3_zslab" | "da3_zslab_global":
                return align_da3_zslab(
                    splat, ref_depth, fx, fy, w, h,
                    self.num_z_slabs, self.MAX_DEPTH, self.smooth_sigma_m,
                )
            case _:  # da3_2dgrid, da3_2dgrid_global
                return align_da3_2dgrid(
                    splat, ref_depth, fx, fy, w, h,
                    self.num_z_slabs, self.num_fov_slabs,
                    self.MAX_DEPTH, self.smooth_sigma_m, self.smooth_sigma_fov,
                )

    def process(
        self,
        views: list[View],
        splats_list: list[Gaussians3D],
        pano_poses: dict = None,
        da3_world_pts=None,
        scale_mode: str = "da3_zslab",
    ) -> tuple[Gaussians3D, dict[int, Gaussians3D]]:
        """Align, trim, pose, and merge Gaussian splats.

        scale_mode:
            'near_edge'        — match nearest-Z across slices (no DA3 required)
            'da3_per_point'    — per-Voronoi-cell DA3 scale
            'da3_zslab'        — Z-depth band DA3 scale
            'da3_zslab_global' — da3_zslab with global slabs across all views
            'da3_2dgrid'       — 2D (Z × FOV) grid DA3 scale
            'da3_2dgrid_global'— da3_2dgrid with global slabs across all views
            'da3_y_ground'     — Y-ground elevation alignment
        """
        # Pre-compute per-view world poses
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

        # Step 1: trim far points (camera space)
        trimmed = [trim_by_max_depth(splat, self.MAX_DEPTH) for splat in splats_list]

        # Step 2: scale alignment
        print(f"--- Scale mode: {scale_mode} ---")
        if scale_mode == "near_edge":
            trimmed = align_near_edge(views, trimmed)

        elif scale_mode in (
            "da3_per_point", "da3_zslab", "da3_zslab_global", "da3_2dgrid", "da3_2dgrid_global"
        ):
            all_da3_pts = None
            if scale_mode in ("da3_zslab_global", "da3_2dgrid_global") and isinstance(da3_world_pts, dict):
                parts = [pts for pts in da3_world_pts.values() if pts is not None]
                all_da3_pts = np.concatenate(parts, axis=0) if parts else None

            for i, (view, splat) in enumerate(zip(views, trimmed)):
                _, center, _, R_c2w = view_poses[i]
                ref_depth = self._resolve_ref_depth(
                    view, center, R_c2w, scale_mode, da3_world_pts, all_da3_pts
                )
                if ref_depth is None:
                    continue
                trimmed[i] = self._align_splat(splat, ref_depth, view, scale_mode)

        elif scale_mode == "da3_y_ground":
            for i, (view, splat) in enumerate(zip(views, trimmed)):
                pano_data = pano_poses.get(view.pano_id) if pano_poses else None
                if pano_data is None:
                    continue
                pts = da3_world_pts.get(view.pano_id) if isinstance(da3_world_pts, dict) else None
                if pts is None:
                    continue
                pts_cam = (pano_data["rotation"] @ (pts - pano_data["center"]).T).T
                da3_elev = elevation_estimate(pts_cam[:, 1], pts_cam[:, 2])
                if da3_elev is not None and da3_elev > 1e-6:
                    trimmed[i] = align_da3_y_ground(splat, da3_elev)

        # Step 3: trim FOV edges, apply world pose
        processed_splats = []
        for i, (view, splat) in enumerate(zip(views, trimmed)):
            _, center, _, R_c2w = view_poses[i]
            splat = trim_by_fov(splat, hfov_limit=view.hfov - 6.0)
            splat = trim_by_max_depth(splat, self.MAX_DEPTH)

            if center is not None:
                c2w = np.eye(4)
                c2w[:3, :3] = R_c2w
                c2w[:3, 3] = center
                splat = apply_transform(
                    splat,
                    torch.tensor(c2w[:3, :], dtype=torch.float32, device=splat.mean_vectors.device),
                )
            else:
                splat = rotate_to_pose(splat, yaw=view.yaw, pitch=view.pitch)

            processed_splats.append((view.pano_id, splat))

        per_pano_splats: dict[int, list] = {}
        for pano_id, splat in processed_splats:
            per_pano_splats.setdefault(pano_id, []).append(splat)
        per_pano_merged = {pid: merge(splats) for pid, splats in per_pano_splats.items()}

        # Inter-pano Voronoi trim: keep each Gaussian only if it's closest (XZ) to its own pano.
        if pano_poses and len(per_pano_merged) > 1:
            print("--- Step: Inter-pano Voronoi trim ---")
            pano_centers = {
                pid: pano_poses[pid]["center"]
                for pid in per_pano_merged
                if pano_poses.get(pid) is not None
            }
            if len(pano_centers) > 1:
                for pid in list(per_pano_merged.keys()):
                    if pid not in pano_centers:
                        continue
                    own = pano_centers[pid]
                    others = [c for p, c in pano_centers.items() if p != pid]
                    per_pano_merged[pid] = trim_by_pano_voronoi(
                        per_pano_merged[pid], own, others, self.voronoi_buffer_m
                    )

                # print("--- Step: Inter-pano Seam Correction ---")
                # per_pano_merged = correct_interpano_seams(
                #     per_pano_merged,
                #     pano_centers,
                #     voronoi_buffer_m=self.voronoi_buffer_m,
                #     seam_band_m=self.voronoi_buffer_m * 4,
                # )

        return merge(list(per_pano_merged.values())), per_pano_merged
