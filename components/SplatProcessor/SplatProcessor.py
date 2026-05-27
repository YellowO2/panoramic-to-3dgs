import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform
from datatype import View
from components.SplatProcessor.utils import (
    project_world_cloud_to_view,
    rotate_to_pose,
    trim_by_fov,
    trim_by_cone,
    trim_by_pitch_bottom,
    trim_by_max_depth,
    trim_beyond_depth,
    trim_depth_range,
    split_depth_zones,
    trim_by_pano_voronoi,
    correct_interpano_seams,
    subsample_gaussians,
    merge,
)
from components.SplatProcessor.alignment import (
    align_near_edge,
    align_da3_per_point,
    align_da3_zslab,
    align_da3_2dgrid,
    align_da3_y_ground,
    align_floor_view,
    da3_floor_elevation,
    elevation_estimate,
)


class SplatProcessor:
    def __init__(
        self,
        num_z_slabs: int = 500,
        num_fov_slabs: int = 250,
        smooth_sigma_m: float = 0.5,
        smooth_sigma_fov: float = 0.15,
        voronoi_buffer_m: float = 1.5,
        floor_keep_fraction: float = 0.6,
        min_depth_coverage: float = 1,
        align_depth: float = 10.0,
        near_depth: float = 48.0,
        sky_depth: float = 50.0,
    ):
        """
        Four depth zones:
          ≤ align_depth              : kept + used for DA3 scale alignment
          align_depth → near_depth   : kept, NOT used for alignment
          near_depth  → sky_depth    : dead zone — trimmed (removes dirt Gaussians)
          > sky_depth                : sky — kept, not aligned
        """
        self.num_z_slabs = num_z_slabs
        self.num_fov_slabs = num_fov_slabs
        self.smooth_sigma_m = smooth_sigma_m
        self.smooth_sigma_fov = smooth_sigma_fov
        self.voronoi_buffer_m = voronoi_buffer_m
        self.floor_keep_fraction = floor_keep_fraction
        self.min_depth_coverage = min_depth_coverage
        self.align_depth = align_depth
        self.near_depth = near_depth
        self.sky_depth = sky_depth

    def _depth_is_sufficient(self, ref_depth: np.ndarray, view: View) -> bool:
        n_valid = int((ref_depth > 0).sum())
        # Need enough pts to meaningfully populate the alignment grid, not proportional to image size
        min_required = max(
            16, int(self.num_z_slabs * self.num_fov_slabs * self.min_depth_coverage)
        )
        print(f"  [Depth coverage] {n_valid} pts (min {min_required})")
        return n_valid >= min_required

    def _try_yground_fallback(
        self, splat, view, pano_poses, da3_world_pts, R_c2w
    ) -> Gaussians3D:
        pano_data = pano_poses.get(view.pano_id) if pano_poses else None
        if pano_data is None:
            return splat
        pts = (
            da3_world_pts.get(view.pano_id)
            if isinstance(da3_world_pts, dict)
            else da3_world_pts
        )
        if pts is None:
            return splat
        # Slice-camera frame: Y is reliably down for pitch=0 views.
        pts_cam = (R_c2w.T @ (pts - pano_data["center"]).T).T
        da3_elev = elevation_estimate(pts_cam[:, 1], pts_cam[:, 2])
        if da3_elev is not None and da3_elev > 1e-6:
            print(
                f"  [Fallback] yaw={view.yaw:.0f}° sparse depth → y_ground (elev={da3_elev:.3f})"
            )
            return align_da3_y_ground(splat, da3_elev)
        return splat

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
                pts, center, R_c2w, view, max_depth=self.align_depth * 1.5
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
                    splat,
                    ref_depth,
                    fx,
                    fy,
                    w,
                    h,
                    self.num_z_slabs,
                    self.align_depth,
                    self.smooth_sigma_m,
                )
            case _:  # da3_2dgrid, da3_2dgrid_global
                return align_da3_2dgrid(
                    splat,
                    ref_depth,
                    fx,
                    fy,
                    w,
                    h,
                    self.num_z_slabs,
                    self.num_fov_slabs,
                    self.align_depth,
                    self.smooth_sigma_m,
                    self.smooth_sigma_fov,
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

        # Pre-compute global DA3 cloud (used by floor alignment and y_ground).
        all_da3_pts = None
        if isinstance(da3_world_pts, dict):
            parts = [pts for pts in da3_world_pts.values() if pts is not None]
            all_da3_pts = np.concatenate(parts, axis=0) if parts else None
        elif da3_world_pts is not None:
            all_da3_pts = da3_world_pts

        # y_ground produces one uniform scalar per slice; it must scale the WHOLE
        # slice rigidly. Apply before split_depth_zones so keep/sky zones ride the
        # same scalar instead of staying at SHARP raw scale (which would create a
        # discontinuity at radial=align_depth). Grid/zslab modes are per-region,
        # so they correctly apply only to the trimmed zone after the split.
        if scale_mode == "da3_y_ground":
            print(f"--- Scale mode: {scale_mode} (pre-split uniform scaling) ---")
            # Per-pano DA3 floor elevation, from each pano's pitch=-90 view.
            # Uses the same concatenated cloud as align_floor_view so the y_ground
            # target matches the floor's target exactly.
            pano_floor_elev: dict = {}
            if all_da3_pts is not None:
                for i, view in enumerate(views):
                    if view.pitch != -90:
                        continue
                    _, center, _, R_c2w = view_poses[i]
                    if center is None:
                        print(f"  [Y-ground] pano {view.pano_id} floor view has no center, skip")
                        continue
                    elev = da3_floor_elevation(view, center, R_c2w, all_da3_pts)
                    if elev is None or elev <= 1e-6:
                        print(f"  [Y-ground] pano {view.pano_id} da3_floor_elevation returned {elev}")
                        continue
                    pano_floor_elev[view.pano_id] = elev
                    print(f"  [Y-ground] pano {view.pano_id} DA3 floor elev: {elev:.4f}")

            for i, (view, splat) in enumerate(zip(views, splats_list)):
                if view.pitch in (90, -90):
                    print(f"  [Y-ground] yaw={view.yaw:+.0f}° pitch={view.pitch:+.0f}° skipped")
                    continue
                da3_elev = pano_floor_elev.get(view.pano_id)
                if da3_elev is None:
                    print(f"  [Y-ground] yaw={view.yaw:+.0f}° pitch={view.pitch:+.0f}° no floor elev for pano {view.pano_id}, skip")
                    continue
                splats_list[i] = align_da3_y_ground(splat, da3_elev)

        # Step 1: single-pass split into three zones.
        #   trimmed     : ≤ align_depth            — used for DA3 scale alignment
        #   keep_splats : align_depth → near_depth — kept but skips alignment
        #   dead zone   : near_depth  → sky_depth  — dropped (dirt Gaussians)
        #   sky_splats  : > sky_depth              — kept, skips alignment
        zones = [
            split_depth_zones(s, self.align_depth, self.near_depth, self.sky_depth)
            for s in splats_list
        ]
        trimmed = [z[0] for z in zones]
        keep_splats = [z[1] for z in zones]
        sky_splats = [z[2] for z in zones]

        # Step 2: scale alignment (near geometry only — trimmed contains no sky)
        print(f"--- Scale mode: {scale_mode} ---")
        if scale_mode == "near_edge":
            trimmed = align_near_edge(views, trimmed)

        elif scale_mode in (
            "da3_per_point",
            "da3_zslab",
            "da3_zslab_global",
            "da3_2dgrid",
            "da3_2dgrid_global",
        ):
            for i, (view, splat) in enumerate(zip(views, trimmed)):
                if view.pitch == -90:
                    continue  # handled by floor alignment step below
                _, center, _, R_c2w = view_poses[i]
                ref_depth = self._resolve_ref_depth(
                    view, center, R_c2w, scale_mode, da3_world_pts, all_da3_pts
                )
                if ref_depth is not None and self._depth_is_sufficient(ref_depth, view):
                    trimmed[i] = self._align_splat(splat, ref_depth, view, scale_mode)
                else:
                    trimmed[i] = self._try_yground_fallback(
                        splat, view, pano_poses, da3_world_pts, R_c2w
                    )

        # Step 2b: floor alignment (-90° pitch), always applied when global DA3 pts are available
        if all_da3_pts is not None:
            for i, (view, splat) in enumerate(zip(views, trimmed)):
                if view.pitch != -90:
                    continue
                _, center, _, R_c2w = view_poses[i]
                if center is None:
                    continue
                print(f"--- Floor alignment: pano {view.pano_id} ---")
                # max_depth bounded independent of align_depth: floor plane fit only
                # makes sense over a short downward range. Large values let far non-floor
                # points dominate the LSQ plane fit and warp the result.
                trimmed[i] = align_floor_view(
                    splat,
                    view,
                    all_da3_pts,
                    center,
                    R_c2w,
                    max_depth=10.0,
                    smooth_sigma_frac=self.smooth_sigma_fov,
                )

        # Re-attach keep and sky — all three go through world-pose transform in step 3
        trimmed = [
            merge([aligned, keep, sky])
            for aligned, keep, sky in zip(trimmed, keep_splats, sky_splats)
        ]

        # Step 3: trim FOV edges, apply world pose
        processed_splats = []
        for i, (view, splat) in enumerate(zip(views, trimmed)):
            _, center, _, R_c2w = view_poses[i]
            if view.pitch == -90:
                splat = trim_by_cone(splat, half_angle_deg=view.hfov / 2.0 - 1.0)
                splat = subsample_gaussians(
                    splat, keep_fraction=self.floor_keep_fraction
                )
            else:
                splat = trim_by_fov(splat, hfov_limit=view.hfov - 6.0)
                if view.pitch == 0:
                    splat = trim_by_pitch_bottom(
                        splat, max_down_deg=view.vfov / 2.0 - 1.0
                    )
            # splat = trim_by_max_depth(splat, self.MAX_DEPTH)

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

            # Move to CPU immediately to avoid accumulating GPU tensors across views
            splat = Gaussians3D(
                mean_vectors=splat.mean_vectors.cpu(),
                singular_values=splat.singular_values.cpu(),
                quaternions=splat.quaternions.cpu(),
                colors=splat.colors.cpu(),
                opacities=splat.opacities.cpu(),
            )
            processed_splats.append((view.pano_id, splat))

        per_pano_splats: dict[int, list] = {}
        for pano_id, splat in processed_splats:
            per_pano_splats.setdefault(pano_id, []).append(splat)
        per_pano_merged = {
            pid: merge(splats) for pid, splats in per_pano_splats.items()
        }

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
