# Panoramic to 3DGS

Convert nearby Google Street View-style panoramas (equirectangular JPEGs) into a merged 3D Gaussian Splat scene for novel view synthesis and immersive visualisation.

## How it works

Each panorama is sliced into perspective views. Apple SHARP turns each view into a set of 3D Gaussians. Depth Anything 3 (DA3) estimates metric depth and camera pose across all panoramas, which is used to scale-align and globally position those Gaussians. The result is a set of PLY files that can be loaded together in any 3DGS viewer.

```
Input: N equirectangular panoramas + metadata.json (lat/lon/heading)
       ↓
  [SpatialGrouper]
  - Orders panos by GPS proximity (greedy nearest-neighbour)
  - Groups into overlapping batches of 3 for DA3

  [ViewExtractor]
  - 6 SHARP views per pano  (horizon + floor, wide FOV)
  - 18 DA3 views per pano   (horizon, 16:9, 20° stride)

  [DA3Model]  — run in batches of 3 panos
  - Multi-view metric depth + camera pose
  - Filters jittery/inconsistent views
  - Output: per-pano center + rotation, depth point cloud
  - Results cached to disk so re-runs skip the model

  [BatchAligner]
  - Each DA3 batch shares 1 pano with the previous batch (bridge pano)
  - Computes rigid transform aligning each batch's local frame to a global world frame
  - Accumulates world-frame poses across all batches

  [SplatGenerator]
  - Apple SHARP: RGB → Gaussians3D per view

  [SplatProcessor]  — run once per DA3 batch (~3 panos at a time)
  - Scale-aligns Gaussians to DA3 depth using a 2D (depth × FOV) grid
  - Special floor alignment for downward-facing views
  - Applies global world poses (places each pano's Gaussians in world space)
  - Trims by FOV, dead-zone (removes noisy mid-range Gaussians), keeps sky
  - Inter-pano Voronoi trim: each Gaussian kept only if closest to its own pano
       ↓
Output: group_{g}_batch_{b}.ply files  (one per DA3 batch, all in world space)
```

## Depth zones

Each Gaussian is categorised by radial depth from the camera:

| Zone | Range | Treatment |
|------|-------|-----------|
| Align | ≤ 10 m | Kept + used for DA3 scale alignment |
| Near-keep | 10 – 48 m | Kept, not aligned |
| Dead zone | 48 – 50 m | Dropped (noisy mid-range Gaussians) |
| Sky | > 50 m | Kept, not aligned |

## Architecture

```
panoramic-to-3dgs/
├── pipeline_batched.py          # Main entry point for multi-pano runs
├── main.py                      # Legacy single-group pipeline
├── datatype.py                  # View dataclass
└── components/
    ├── SpatialGrouper.py        # GPS ordering + batch construction
    ├── BatchAligner.py          # Rigid transform between DA3 local frames
    ├── ViewExtractor/           # Equirectangular → perspective slices
    ├── DepthMapGenerator/
    │   ├── DA3Model.py          # Multi-view depth + pose (main)
    │   └── DA360DepthModel.py   # Single-pano depth (legacy)
    ├── SplatGenerator/          # Wraps Apple SHARP inference
    ├── SplatProcessor/
    │   ├── SplatProcessor.py    # Orchestrates alignment, trimming, merging
    │   ├── alignment.py         # Scale alignment algorithms (2dgrid, zslab, y-ground)
    │   └── utils.py             # Depth trimming, Voronoi, subsampling helpers
    └── ImageCleaner/            # Optional: remove people via inpainting
```

## Installation

```bash
# Install as a package (for use in other projects)
pip install git+https://github.com/YellowO2/panoramic-to-3dgs.git

# Or clone and install locally
git clone https://github.com/YellowO2/panoramic-to-3dgs.git
cd panoramic-to-3dgs
pip install -r requirements.txt

# SHARP model
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt -P ./models/

# DA3 model
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --local-dir ./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

## Input format

```
data/inputs/my_location/
├── metadata.json          # [{id, lat, lon, heading}, ...]
├── pano_{id}.jpg
├── pano_{id}.jpg
└── ...
```

`metadata.json` can be a flat list `[{id, lat, lon, heading}, ...]` or a dict `{nodes: [...], spatial_order: [...]}`.

## Running

Edit the paths at the bottom of `pipeline_batched.py` and run:

```bash
python pipeline_batched.py
```

Key parameters in `run_batched_pipeline`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 3 | Panos per DA3 call (VRAM limited) |
| `batches_per_group` | 3 | DA3 batches before saving one set of PLYs |
| `scale_mode` | `da3_2dgrid_global` | Gaussian scale alignment strategy |
| `sharp_subbatch_size` | 4 | Panos processed before offloading Gaussians to CPU |

## Output

```
output_dir/
├── views_{i}_da3/                # DA3 perspective slices (cached)
├── views_{i}_sharp/              # SHARP perspective slices
├── da3_cache_{i}_{j}_{k}.npz    # Cached DA3 poses + depth pts per batch
├── group_0_batch_0.ply           # 3DGS output — load all PLYs together
├── group_0_batch_1.ply
├── group_0_batch_2.ply
├── group_0_metadata.json
└── ...
```

All PLY files share the same world coordinate frame and can be loaded simultaneously in any 3DGS viewer (e.g. SuperSplat, Gaussian Splatting WebGL viewer).
