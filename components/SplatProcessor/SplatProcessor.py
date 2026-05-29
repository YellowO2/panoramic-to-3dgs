import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform
from datatype import View
from components.SplatProcessor.utils import (
    project_world_cloud_to_view,
    rotate_to_pose,
    scale_gaussians,
    trim_by_fov,
    trim_by_cone,
    trim_by_pitch_bottom,
    split_depth_zones,
    subsample_gaussians,
    merge,
)
from components.SplatProcessor.alignment import (
    align_near_edge,
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
        self.floor_keep_fraction = floor_keep_fraction
        self.min_depth_coverage = min_depth_coverage
        self.align_depth = align_depth
        self.near_depth = near_depth
        self.sky_depth = sky_depth

    def _depth_is_sufficient(self, ref_depth: np.ndarray) -> bool:
        n_valid = int((ref_depth > 0).sum())
        # Need enough pts to meaningfully populate the alignment grid, not proportional to image size
        min_required = max(
            16, int(self.num_z_slabs * self.num_fov_slabs * self.min_depth_coverage)
        )
        print(f"  [Depth coverage] {n_valid} pts (min {min_required})")
        return n_valid >= min_required

    def _try_yground_fallback(
        self, splat, view, pano_poses, all_da3_pts, R_c2w
    ) -> Gaussians3D:
        # Y-ground elevation only makes sense when slice-camera Y is gravity-down,
        # which is true for pitch=0 side slices but not for pitch=±90.
        if view.pitch in (90, -90):
            print(f"  [Fallback] yaw={view.yaw:.0f}° pitch={view.pitch:+.0f}° skipped (not gravity-aligned)")
            return splat
        pano_data = pano_poses.get(view.pano_id) if pano_poses else None
        if pano_data is None or all_da3_pts is None:
            return splat
        # Slice-camera frame: Y is reliably down for pitch=0 views.
        pts_cam = (R_c2w.T @ (all_da3_pts - pano_data["center"]).T).T
        da3_elev = elevation_estimate(pts_cam[:, 1], pts_cam[:, 2])
        if da3_elev is not None and da3_elev > 1e-6:
            print(
                f"  [Fallback] yaw={view.yaw:.0f}° sparse depth → y_ground (elev={da3_elev:.3f})"
            )
            return align_da3_y_ground(splat, da3_elev)
        return splat

    def process(
        self,
        views: list[View],
        splats_list: list[Gaussians3D],
        pano_poses: dict,
        all_da3_pts: np.ndarray | None,
        scale_mode: str = "da3_2dgrid_global",
        n_da3_clean: int | None = None,
        target_pano_id: int = 0,
    ) -> Gaussians3D:
        """Align, trim, pose, and merge Gaussian splats for the target pano.

        scale_mode:
            'da3_2dgrid_global' — 2D (Z × FOV) grid DA3 scale (recommended)
            'da3_y_ground'      — Y-ground elevation alignment (uniform per slice)
            'near_edge'         — match nearest-Z across slices (no depth model required)

        Returns the merged splat, anchored so the target pano's capture point
        lands at world (0,0,0).
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

        # SHARP-only fallback: DA3 cloud is unreliable (too few cleaned slices),
        # so ignore DA3 entirely and align every slice to the median SHARP
        # elevation across side slices.
        sharp_only_fallback = (
            scale_mode.startswith("da3_")
            and n_da3_clean is not None
            and n_da3_clean < 6
        )
        if sharp_only_fallback:
            print(
                f"--- DA3 cloud unreliable ({n_da3_clean} clean slices < 6), "
                f"SHARP-only fallback ---"
            )
            sharp_elevs = []
            for view, splat in zip(views, splats_list):
                if view.pitch in (90, -90):
                    continue
                mv = splat.mean_vectors[0].detach().cpu().numpy()
                e = elevation_estimate(mv[:, 1], mv[:, 2])
                if e is not None and e > 1e-6:
                    sharp_elevs.append(e)
            target = float(np.median(sharp_elevs)) if sharp_elevs else None
            print(
                f"  [SHARP-only] side-slice elevs: "
                f"{[f'{e:.3f}' for e in sharp_elevs]}  target (median): "
                f"{target if target is None else f'{target:.4f}'}"
            )

            if target is not None:
                for i, (view, splat) in enumerate(zip(views, splats_list)):
                    if view.pitch == 90:
                        print(f"  [SHARP-only] yaw={view.yaw:+.0f}° sky skipped")
                        continue
                    if view.pitch == -90:
                        # Floor: scale by target / median Z (camera-above-floor distance).
                        mv = splat.mean_vectors[0].detach().cpu().numpy()
                        forward = mv[:, 2] > 0
                        if not forward.any():
                            continue
                        sharp_z = float(np.median(mv[forward, 2]))
                        if sharp_z <= 1e-6:
                            continue
                        scale = target / sharp_z
                        print(f"  [SHARP-only] floor SHARP Z: {sharp_z:.4f}  scale: {scale:.4f}")
                        splats_list[i] = scale_gaussians(splat, scale)
                    else:
                        splats_list[i] = align_da3_y_ground(splat, target)

            # Disable downstream DA3 paths.
            all_da3_pts = None
            scale_mode = "_sharp_only"

        # y_ground produces one uniform scalar per slice; it must scale the WHOLE
        # slice rigidly. Apply before split_depth_zones so keep/sky zones ride the
        # same scalar instead of staying at SHARP raw scale (which would create a
        # discontinuity at radial=align_depth). Grid mode is per-region,
        # so it correctly applies only to the trimmed zone after the split.
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

        elif scale_mode == "da3_2dgrid_global":
            for i, (view, splat) in enumerate(zip(views, trimmed)):
                if view.pitch == -90:
                    continue  # handled by floor alignment step below
                _, center, _, R_c2w = view_poses[i]
                ref_depth = None
                if all_da3_pts is not None and center is not None:
                    ref_depth = project_world_cloud_to_view(
                        all_da3_pts, center, R_c2w, view,
                        max_depth=self.align_depth * 1.5,
                    )
                if ref_depth is not None and self._depth_is_sufficient(ref_depth):
                    trimmed[i] = align_da3_2dgrid(
                        splat,
                        ref_depth,
                        view.focal_px, view.focal_px,
                        int(view.width), int(view.height),
                        self.num_z_slabs,
                        self.num_fov_slabs,
                        self.align_depth,
                        self.smooth_sigma_m,
                        self.smooth_sigma_fov,
                    )
                else:
                    trimmed[i] = self._try_yground_fallback(
                        splat, view, pano_poses, all_da3_pts, R_c2w
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
            processed_splats.append(splat)

        merged = merge(processed_splats)

        # Anchor target pano's capture point at (0,0,0) so viewers that place the
        # camera at world origin land at the capture point.
        anchor = (
            pano_poses[target_pano_id]["center"]
            if pano_poses and target_pano_id in pano_poses
            else None
        )
        if anchor is not None and np.linalg.norm(anchor) > 1e-9:
            print(f"  [Anchor] Shifting cloud so pano {target_pano_id} center {anchor} → (0,0,0)")
            anchor_t = torch.tensor(anchor, dtype=torch.float32)
            merged = Gaussians3D(
                mean_vectors=merged.mean_vectors - anchor_t,
                singular_values=merged.singular_values,
                quaternions=merged.quaternions,
                colors=merged.colors,
                opacities=merged.opacities,
            )

        return merged
