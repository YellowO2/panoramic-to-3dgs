import gc
import json
import os

import numpy as np
import torch
from sharp.utils.gaussians import Gaussians3D, save_ply

from components.BatchAligner import apply_to_pose, apply_to_pts, compute_local_to_world
from components.DepthMapGenerator.DA3Model import DA3Model
from components.SpatialGrouper import make_batches, nearest_neighbor_order
from components.SplatGenerator.SplatGenerator import SplatGenerator
from components.SplatProcessor.SplatProcessor import SplatProcessor
from components.SplatProcessor.utils import backproject_views_to_pcd, merge as merge_gaussians
from components.ViewExtractor.ViewExtractor import extract_views, extract_views_for_da3
from datatype import View


# ── Helpers ───────────────────────────────────────────────────────────────────

def _move_to_cpu(g: Gaussians3D) -> Gaussians3D:
    for attr in dir(g):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(g, attr)
            if isinstance(val, torch.Tensor):
                setattr(g, attr, val.cpu())
        except Exception:
            pass
    return g


def load_metadata(metadata_path: str) -> tuple[dict, list[str]]:
    """
    Load metadata.json.  Accepts two formats:
      - flat list  : [{id, lat, lon, heading, ...}, ...]
      - dict       : {nodes: [...], spatial_order: [...]}  (spatial_order optional)

    Returns (nodes_by_id, ordered_ids).
    """
    with open(metadata_path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        nodes = raw
        ordered_ids = nearest_neighbor_order(nodes)
    else:
        nodes = raw['nodes']
        ordered_ids = raw.get('spatial_order') or nearest_neighbor_order(nodes)

    return {n['id']: n for n in nodes}, ordered_ids


# ── DA3 batch ─────────────────────────────────────────────────────────────────

def _run_da3_batch(
    batch_pano_ids: list[str],
    pano_idx_of: dict[str, int],
    folder_path: str,
    output_dir: str,
    da3_model: DA3Model,
    world_poses: dict[int, dict],   # mutated in place
    group_da3_pts: dict[int, np.ndarray],  # mutated in place
) -> None:
    """
    Run DA3 on one batch of panos, align poses to the global world frame via the
    shared bridge pano, and accumulate world-frame camera poses and depth point clouds.
    Results are cached to disk so subsequent runs skip the model entirely.
    """
    batch_idxs = [pano_idx_of[pid] for pid in batch_pano_ids]
    cache_path = os.path.join(output_dir, f"da3_cache_{'_'.join(str(i) for i in batch_idxs)}.npz")

    # ── Cache hit: load poses and pts, skip model entirely ───────────────────
    if os.path.exists(cache_path):
        print(f"  [DA3 cache] Loading {cache_path}")
        data = np.load(cache_path)
        for key in data.files:
            if key.startswith('pose_center_'):
                idx = int(key[len('pose_center_'):])
                if idx not in world_poses:
                    world_poses[idx] = {
                        'center':   data[f'pose_center_{idx}'],
                        'rotation': data[f'pose_rotation_{idx}'],
                    }
            elif key.startswith('pts_'):
                idx = int(key[len('pts_'):])
                if idx not in group_da3_pts:
                    group_da3_pts[idx] = data[key]
        return

    # ── Cache miss: extract views, run DA3, save cache ────────────────────────
    # 1. Extract DA3 views
    all_views: list[View] = []
    for pid in batch_pano_ids:
        idx = pano_idx_of[pid]
        img_path = os.path.join(folder_path, f"pano_{pid}.jpg")
        da3_dir = os.path.join(output_dir, f"views_{idx}_da3")
        os.makedirs(da3_dir, exist_ok=True)
        all_views.extend(
            extract_views_for_da3(img_path, da3_dir, prefix=f"pano_{idx}_", pano_id=idx)
        )

    # 2. Run DA3
    filtered_views, da3_result = da3_model.process_views(all_views)
    local_poses = da3_result.pano_poses  # {pano_idx: {center, rotation}} in DA3 local frame

    # 3. Determine rigid transform from DA3 local frame → world frame
    # Scan batch panos in order — first one present in both world_poses and local_poses
    # becomes the bridge (handles cases where the designated overlap pano was filtered).
    bridge_idx = None
    for pid in batch_pano_ids:
        idx = pano_idx_of[pid]
        if idx in world_poses and idx in local_poses:
            bridge_idx = idx
            break
    has_bridge = bridge_idx is not None
    print(f"  [Bridge] chosen={bridge_idx} | has_bridge={has_bridge} | local_poses keys={list(local_poses.keys())}")

    if has_bridge:
        R_l2w, t_l2w = compute_local_to_world(
            local_poses[bridge_idx]['rotation'], local_poses[bridge_idx]['center'],
            world_poses[bridge_idx]['rotation'], world_poses[bridge_idx]['center'],
        )
    else:
        R_l2w, t_l2w = np.eye(3), np.zeros(3)

    # 4. Register new panos and collect for cache
    cache_data = {}
    for idx, local_pose in local_poses.items():
        if idx not in world_poses:
            world_poses[idx] = apply_to_pose(local_pose, R_l2w, t_l2w)
            cache_data[f'pose_center_{idx}']   = world_poses[idx]['center']
            cache_data[f'pose_rotation_{idx}'] = world_poses[idx]['rotation']

    # 5. Backproject depth → world-frame point cloud
    _, _, local_pts = backproject_views_to_pcd(filtered_views, da3_result)
    for idx, pts in local_pts.items():
        if idx not in group_da3_pts:
            group_da3_pts[idx] = apply_to_pts(pts, R_l2w, t_l2w)
            cache_data[f'pts_{idx}'] = group_da3_pts[idx]

    if cache_data:
        np.savez(cache_path, **cache_data)
        print(f"  [DA3 cache] Saved {cache_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_batched_pipeline(
    metadata_path: str,
    folder_path: str,
    output_dir: str,
    model_paths: dict,
    batch_size: int = 4,
    batches_per_group: int = 3,
    depth_mode: str = 'da3',
    sharp_subbatch_size: int = 4,
    scale_mode: str = 'da3_2dgrid_global',
) -> None:
    """
    Process N panoramas in batched fashion.

    Each DA3 batch processes `batch_size` panos (sharing 1 with the previous batch
    for world-frame alignment).  Every `batches_per_group` DA3 batches, the
    accumulated splats are merged and saved as one PLY file.

    Args:
        metadata_path     : path to metadata.json (flat list or {nodes, spatial_order})
        folder_path       : directory containing pano_{id}.jpg images
        output_dir        : where to write group PLYs and metadata JSONs
        model_paths       : {'da3': ..., 'sharp': ...}
        batch_size        : panos per DA3 call (default 4, limited by VRAM)
        batches_per_group : DA3 batches before saving one PLY (default 3 → ~10 unique panos)
        depth_mode        : 'da3' or None
        sharp_subbatch_size: panos processed before offloading Gaussians to CPU
        scale_mode        : SplatProcessor scale mode
    """
    nodes_by_id, ordered_ids = load_metadata(metadata_path)
    os.makedirs(output_dir, exist_ok=True)

    pano_idx_of: dict[str, int] = {pid: i for i, pid in enumerate(ordered_ids)}
    batches = make_batches(ordered_ids, batch_size)
    groups = [
        batches[i: i + batches_per_group]
        for i in range(0, len(batches), batches_per_group)
    ]

    print(f"Pipeline: {len(ordered_ids)} panos | {len(batches)} DA3 batches | {len(groups)} output groups")

    world_poses: dict[int, dict] = {}   # pano_idx → {center, rotation} in world frame
    sharp_done: set[str] = set()        # pano_ids already SHARP-processed

    for group_idx, group_batches in enumerate(groups):

        # Unique ordered pano_ids spanning this group
        seen: set[str] = set()
        all_group_ids: list[str] = []
        for batch in group_batches:
            for pid in batch:
                if pid not in seen:
                    seen.add(pid)
                    all_group_ids.append(pid)

        print(f"\n{'='*60}")
        print(f"GROUP {group_idx} | {len(all_group_ids)} unique panos | {len(group_batches)} DA3 batches")

        group_da3_pts: dict[int, np.ndarray] = {}

        # ── Phase 1: DA3 ─────────────────────────────────────────────
        if depth_mode == 'da3':
            da3 = DA3Model(model_paths['da3'])
            for batch_ids in group_batches:
                idxs = [pano_idx_of[p] for p in batch_ids]
                print(f"  DA3 batch → pano indices {idxs}")
                _run_da3_batch(
                    batch_pano_ids=batch_ids,
                    pano_idx_of=pano_idx_of,
                    folder_path=folder_path,
                    output_dir=output_dir,
                    da3_model=da3,
                    world_poses=world_poses,
                    group_da3_pts=group_da3_pts,
                )
            del da3
            torch.cuda.empty_cache()
            gc.collect()

        # ── Phase 2: SHARP ───────────────────────────────────────────
        # Only generate splats for panos not yet processed (bridge pano skipped)
        new_pano_ids = [pid for pid in all_group_ids if pid not in sharp_done]
        print(f"  SHARP: {len(new_pano_ids)} new panos")

        # Index by pano_idx so phase 3 can retrieve per-batch subsets
        pano_to_views:     dict[int, list[View]]       = {}
        pano_to_gaussians: dict[int, list[Gaussians3D]] = {}

        sharp_gen = SplatGenerator(model_paths['sharp'])
        for sub_start in range(0, len(new_pano_ids), sharp_subbatch_size):
            sub_ids = new_pano_ids[sub_start: sub_start + sharp_subbatch_size]
            for pid in sub_ids:
                idx = pano_idx_of[pid]
                img_path = os.path.join(folder_path, f"pano_{pid}.jpg")
                sharp_dir = os.path.join(output_dir, f"views_{idx}_sharp")
                os.makedirs(sharp_dir, exist_ok=True)
                views = extract_views(
                    img_path, sharp_dir,
                    overlap_degrees=10, slice_count=5,
                    prefix=f"pano_{idx}_", pano_id=idx,
                )
                pano_to_views[idx] = []
                pano_to_gaussians[idx] = []
                for v in views:
                    g = sharp_gen.generate_from_view(v)
                    _move_to_cpu(g)
                    pano_to_views[idx].append(v)
                    pano_to_gaussians[idx].append(g)
            torch.cuda.empty_cache()

        del sharp_gen
        torch.cuda.empty_cache()
        gc.collect()
        sharp_done.update(new_pano_ids)

        # ── Phase 3: Process per DA3 batch, then merge + save ────────
        # Each batch processes ~4 panos, limiting peak GPU to ~24 views.
        # Voronoi trim is per-batch only (slight overlap at batch boundaries).
        processor = SplatProcessor()
        partial_splats: list[Gaussians3D] = []
        sharp_processed: set[int] = set()
        first_views: list[View] = []

        for batch_ids in group_batches:
            # Only include panos that are new to this group AND not yet processed
            batch_idxs = [
                pano_idx_of[pid] for pid in batch_ids
                if pano_idx_of[pid] in pano_to_views and pano_idx_of[pid] not in sharp_processed
            ]
            if not batch_idxs:
                continue

            batch_views:     list[View]       = []
            batch_gaussians: list[Gaussians3D] = []
            for idx in batch_idxs:
                batch_views.extend(pano_to_views[idx])
                batch_gaussians.extend(pano_to_gaussians[idx])

            if not first_views:
                first_views = batch_views[:1]

            batch_pano_poses = (
                {idx: world_poses[idx] for idx in batch_idxs if idx in world_poses}
                if depth_mode == 'da3' else None
            )
            batch_da3_pts = (
                {idx: group_da3_pts[idx] for idx in batch_idxs if idx in group_da3_pts}
                if depth_mode == 'da3' else None
            )

            print(f"  Processing DA3 batch: pano indices {batch_idxs}")
            partial, _ = processor.process(
                views=batch_views,
                splats_list=batch_gaussians,
                pano_poses=batch_pano_poses,
                da3_world_pts=batch_da3_pts,
                scale_mode=scale_mode,
            )
            # Save this batch's partial splat immediately
            batch_ply = os.path.join(output_dir, f"group_{group_idx}_batch_{len(partial_splats)}.ply")
            ref = batch_views[0]
            save_ply(partial, f_px=ref.focal_px, image_shape=(ref.height, ref.width), path=batch_ply)
            print(f"  Saved: {batch_ply}")
            partial_splats.append(batch_ply)
            sharp_processed.update(batch_idxs)

            # Free this batch's per-pano data immediately
            for idx in batch_idxs:
                del pano_to_views[idx]
                del pano_to_gaussians[idx]
            del partial
            torch.cuda.empty_cache()
            gc.collect()

        group_meta = {
            'group_id': group_idx,
            'output_plys': partial_splats,
            'pano_count': len(new_pano_ids),
            'panos': [
                {
                    'pano_id': pid,
                    'pano_idx': pano_idx_of[pid],
                    'world_center': world_poses[pano_idx_of[pid]]['center'].tolist()
                        if pano_idx_of[pid] in world_poses else None,
                    'world_rotation': world_poses[pano_idx_of[pid]]['rotation'].tolist()
                        if pano_idx_of[pid] in world_poses else None,
                }
                for pid in new_pano_ids
            ],
        }
        with open(os.path.join(output_dir, f"group_{group_idx}_metadata.json"), 'w') as f:
            json.dump(group_meta, f, indent=2)

        # Clear this group's data before starting the next
        del pano_to_views, pano_to_gaussians, partial_splats, group_da3_pts
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone. {len(groups)} group PLYs saved to: {output_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    models = {
        'da3': 'models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7',
        'sharp': 'models/sharp_2572gikvuh.pt',
    }
    run_batched_pipeline(
        # metadata_path='data/inputs/panoramas_1777658753296/metadata.json',
        # folder_path='data/inputs/panoramas_1777658753296',
             metadata_path='data/inputs/panoramas_large/metadata.json',
        folder_path='data/inputs/panoramas_large',
        output_dir='data/outputs/batched_test',
        model_paths=models,
    )
