# Panoramic to 3DGS

Convert nearby Google Street View-style panoramas (equirectangular JPEGs) into a merged 3D Gaussian Splat scene for novel view synthesis and immersive visualisation.

## How it works

Each panorama is sliced into perspective views. Apple SHARP turns each view into a set of 3D Gaussians. Depth Anything 3 (DA3) jointly estimates metric depth and camera pose across all panoramas, which is used to scale-align and globally position those Gaussians. The result is a merged PLY file (plus per-pano PLYs) that can be loaded in any 3DGS viewer.

```
Input: N equirectangular panoramas
       ↓
  [ViewExtractor]
  - SHARP views per pano   (horizon + floor, wide FOV)
  - 18 DA3 views per pano  (horizon, 16:9, 20° stride)

  [DA3Model]  — one joint call over all panos
  - Multi-view metric depth + camera pose
  - Filters jittery views; snaps to consensus per-pano pose
  - Output: per-pano center + rotation, depth point cloud per pano

  [SplatGenerator]
  - Apple SHARP: RGB → Gaussians3D per view

  [SplatProcessor]
  - Scale-aligns Gaussians to DA3 depth using a 2D (depth × FOV) grid
  - Special floor alignment for downward-facing views
  - Applies per-pano world poses (places Gaussians in world space)
  - Trims by FOV, dead-zone (drops noisy mid-range Gaussians), keeps sky
  - Inter-pano Voronoi trim: each Gaussian kept only if closest to its own pano
       ↓
Output: final_output.ply (merged) + output_pano_{i}.ply (per pano)
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
├── main.py                      # Example entry point
├── config.yaml                  # Pipeline config (depth mode, scale mode, thresholds)
├── datatype.py                  # View dataclass
├── panoramic_to_3dgs/
│   ├── pipeline.py              # Pipeline class (main API)
│   └── config.py                # PipelineConfig
└── components/
    ├── ViewExtractor/           # Equirectangular → perspective slices
    ├── DepthMapGenerator/
    │   ├── DA3Model.py          # Multi-view depth + pose (main)
    │   └── DA360DepthModel.py   # Single-pano depth (legacy)
    ├── SplatGenerator/          # Wraps Apple SHARP inference
    ├── SplatProcessor/
    │   ├── SplatProcessor.py    # Orchestrates alignment, trimming, merging
    │   ├── alignment.py         # Scale alignment algorithms (2dgrid, zslab, y-ground)
    │   └── utils.py             # Depth trimming, Voronoi, subsampling helpers
    ├── ImageCleaner/            # Optional: remove people via inpainting
    └── Saver/                   # Debug PLY/PCD writing
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

## Running

```python
from panoramic_to_3dgs import Pipeline, PipelineConfig

config = PipelineConfig.from_yaml("config.yaml")
pipeline = Pipeline(config)

pipeline.run(
    panorama_paths=["data/inputs/scene/pano_A.jpg", "data/inputs/scene/pano_B.jpg"],
    output_dir="data/outputs/scene",
)
```

Or run the example entry point: `python main.py`.

### Support panoramas (depth-only context)

Pass extra nearby panoramas that should contribute to DA3 depth/pose estimation but **not** be turned into Gaussians. They add translation baselines so target depth is better aligned.

```python
pipeline.run(
    panorama_paths=[target, support_1, support_2, support_3],
    generate_pano_ids=[0],   # only the target gets SHARP'd
    output_dir="data/outputs/scene",
)
```

DA3 sees all 4 panos jointly; SHARP only runs on the target.

## Input format

Panoramas are equirectangular JPEGs. Paths are passed directly to `Pipeline.run`. There is also a `load_panorama_folder(folder)` helper that reads:

```
folder/
├── metadata.json   # [{id, lat, lon, heading, ...}, ...]
└── pano_{id}.jpg
```

and returns the list of paths.

## Output

```
output_dir/
├── final_output.ply          # Merged scene (all generated panos)
├── output_pano_{i}.ply       # Per-pano splats (one per generated pano)
└── (debug artifacts when config.debug = true)
```
