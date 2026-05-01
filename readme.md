# Panoramic to 3DGS

## TODO:
- Allow lower pitch for the da3 view extracting process
- Allow pipeline of large amounts of panorama like 30 panoramas. This means saving per pano, and processing da3 such that common points it shared.

## Project Goal
Convert nearby Google Street View-style panoramas (equirectangular) into a merged 3D Gaussian Splat scene for novel view synthesis and immersive visualisation.

The pipeline uses Apple SHARP to generate per-view Gaussians, and Depth Anything 3 (DA3) to produce a globally consistent point cloud that is used to clean and align those Gaussians.

## Current Focus
Getting DA3 to produce a reliable global point cloud across multiple panoramas. The DA3 outputs have pose drift and jitter across panorama slices, so `DA3Model` filters them — but that filtering logic is still being revised.

## Big Picture Pipeline
```
Input: N nearby equirectangular panoramas
       ↓
  [ViewExtractor]
  ├─ 6 SHARP views per pano  (4 horizon + 2 poles, wide FOV)
  └─ 8 DA3 views per pano    (horizon only, 16:9, 50% overlap)
       ↓
  [DA3Model]  ← ACTIVE WORK
  - Multi-view depth + camera pose inference
  - Filter jittery/drifting predictions per pano
  - Output: per-pano consensus center + rotation, depth maps
       ↓
  [SplatGenerator]
  - Apple SHARP: RGB → Gaussians3D per view
       ↓
  [SplatProcessor]
  - Align Gaussians to DA3 depth (scale correction)
  - Apply DA3 global poses
  - Trim by FOV, merge all splats
       ↓
Output: final_output.ply  (merged 3DGS)
```

## Architecture Summary
- `datatype.py` — `View` dataclass: image path, yaw/pitch/FOV, focal length, pano_id, optional depth + splat
- `components/ViewExtractor/` — slices equirectangular panos into perspective `View` objects (two strategies)
- `components/DepthMapGenerator/`
  - `DA360DepthModel` — panorama-level depth (single-pano, faster)
  - `DA3Model` — multi-view depth + metric pose (multi-pano, current focus)
- `components/SplatGenerator/` — wraps Apple SHARP inference
- `components/SplatProcessor/` — model-free: depth alignment, FOV trimming, pose application, merging
- `components/ImageCleaner/` — optional: removes people/objects via diffusion inpainting
- `components/Saver/` — writes PLY point clouds and depth images
- `main.py` — orchestrator; `depth_mode` toggle selects depth strategy (`'da3'`, `'da360'`, `'external'`, `None`)

## Installation
```bash
pip install -r requirements.txt

# SHARP model
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt -P ./models/

# DA3 model
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir ./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

## Running
Configure paths and panorama inputs in `main.py`, then:
```bash
python main.py
```

Key `run_panoramic_pipeline` parameters:
- `panorama_paths` — list of input pano image paths
- `output_dir` — where to write outputs
- `depth_mode` — `'da3'` (recommended), `'da360'`, `'external'`, or `None`
- `model_paths` — dict with keys `'da360'`, `'da3'`, `'sharp'`

## Output Structure
```
output_dir/
├── views_pano_i_sharp/       # 6 SHARP perspective slices
├── views_pano_i_da3/         # 8 DA3 perspective slices
├── gs/                        # per-view intermediate PLY files
└── final_output.ply           # merged 3D Gaussian splat
```

## Manual DA3 CLI (reference)
```bash
da3 auto ./all_views \
  --export-format glb \
  --export-dir ./output_depth \
  --model-dir ./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7
```
