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
    merge,
)
from components.SplatProcessor.alignment import (
    align_near_edge,
    align_da3_per_point,
    align_da3_zslab,
    align_da3_y_ground,
    elevation_estimate,
)


class SplatProcessor:
    MAX_DEPTH = 10.0

    def __init__(self, num_z_slabs: int = 500, smooth_sigma_m: float = 0.5):
        self.num_z_slabs = num_z_slabs
        self.smooth_sigma_m = smooth_sigma_m

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
            'near_edge'     — match nearest-Z across slices (no DA3 required)
            'da3_per_point' — per-Voronoi-cell DA3 scale
            'da3_zslab'     — Z-depth band DA3 scale (slicing on z axis)
            'da3_y_ground'  — Y-ground elevation alignment
            'da3_zslab_global' — like da3_zslab but with global Z slabs computed from all views combined
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
        elif scale_mode in ("da3_per_point", "da3_zslab", "da3_zslab_global"):
            all_da3_pts = None
            if scale_mode == "da3_zslab_global" and isinstance(da3_world_pts, dict):
                parts = [pts for pts in da3_world_pts.values() if pts is not None]
                all_da3_pts = np.concatenate(parts, axis=0) if parts else None

            for i, (view, splat) in enumerate(zip(views, trimmed)):
                _, center, _, R_c2w = view_poses[i]
                if scale_mode == "da3_zslab_global":
                    pts = all_da3_pts
                else:
                    pts = (
                        da3_world_pts.get(view.pano_id)
                        if isinstance(da3_world_pts, dict)
                        else da3_world_pts
                    )
                ref_depth = None
                if pts is not None and center is not None:
                    ref_depth = project_world_cloud_to_view(pts, center, R_c2w, view)
                elif view.depth is not None:
                    ref_depth = view.depth

                if ref_depth is None:
                    continue

                if scale_mode == "da3_per_point":
                    trimmed[i] = align_da3_per_point(
                        splat,
                        ref_depth,
                        view.focal_px,
                        view.focal_px,
                        int(view.width),
                        int(view.height),
                    )
                else:
                    trimmed[i] = align_da3_zslab(
                        splat,
                        ref_depth,
                        view.focal_px,
                        view.focal_px,
                        int(view.width),
                        int(view.height),
                        self.num_z_slabs,
                        self.MAX_DEPTH,
                        self.smooth_sigma_m,
                    )
        elif scale_mode == "da3_y_ground":
            # Pre-compute DA3 elevation target once per panorama.
            # For pitch=0 slices, yaw rotation doesn't affect Y, so any R_c2w works —
            # we use pano_rot.T (equivalent to yaw=0 slice).
            da3_elev_per_pano: dict[int, float] = {}
            if isinstance(da3_world_pts, dict) and pano_poses:
                for pano_id, world_pts in da3_world_pts.items():
                    pano_data = pano_poses.get(pano_id)
                    if pano_data is None:
                        continue
                    center = pano_data["center"]
                    pano_rot = pano_data["rotation"]
                    R_w2c = pano_rot  # R_w2c = (pano_rot.T).T = pano_rot
                    pts_cam = (R_w2c @ (world_pts - center).T).T
                    elev = elevation_estimate(pts_cam[:, 1], pts_cam[:, 2])
                    if elev is not None and elev > 1e-6:
                        da3_elev_per_pano[pano_id] = elev
                        print(
                            f"  [Y-ground] Pano {pano_id} DA3 elevation target: {elev:.4f}"
                        )
                    else:
                        print(
                            f"  [Y-ground] Pano {pano_id} DA3 elevation invalid, will skip."
                        )

            for i, (view, splat) in enumerate(zip(views, trimmed)):
                da3_elev = da3_elev_per_pano.get(view.pano_id)
                if da3_elev is None:
                    continue
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
                    torch.tensor(
                        c2w[:3, :],
                        dtype=torch.float32,
                        device=splat.mean_vectors.device,
                    ),
                )
            else:
                splat = rotate_to_pose(splat, yaw=view.yaw, pitch=view.pitch)

            processed_splats.append((view.pano_id, splat))

        per_pano_splats: dict[int, list] = {}
        for pano_id, splat in processed_splats:
            per_pano_splats.setdefault(pano_id, []).append(splat)
        per_pano_merged = {
            pid: merge(splats) for pid, splats in per_pano_splats.items()
        }
        return merge([s for _, s in processed_splats]), per_pano_merged
