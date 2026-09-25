# Panoramic to 3DGS

Turn one equirectangular panorama into a 3D Gaussian Splat. Depth comes from outside: the panorama's pose and a DA3 point cloud around it, which [streetview-to-3d](https://github.com/YellowO2/streetview-to-3d) makes from the panorama and its same-capture neighbours (`da3_ops.depth_around`). This package only makes and aligns the splat, and does not need DA3 installed.

## How it works

```
Input: one panorama, plus depth (points + the panorama's pose) from outside
       ↓
  [ViewExtractor]
  - Cuts the panorama into perspective slices for SHARP.

  [SplatGenerator]
  - Apple SHARP: RGB -> Gaussians3D per slice

  [SplatProcessor]
  - Scale-aligns each slice's Gaussians to the depth (2dgrid or y-ground)
  - Special floor alignment for the downward-facing view
  - Applies the pose, trims by FOV, drops a noisy mid-range zone, keeps sky
  - Anchors the panorama's capture point to (0, 0, 0)
       ↓
Output: final_output.ply
```

## Scale modes

| Mode | Description |
|------|-------------|
| `da3_2dgrid_global` *(recommended)* | Scale each Gaussian by the median DA3/SHARP ratio inside a 2D (depth × FOV) cell, then smooth across cells. |
| `da3_y_ground` | Uniform per-slice scale so SHARP's ground elevation matches DA3's ground elevation. Simpler and more robust when depth coverage is patchy. |
| `near_edge` | Match nearest-Z across slices (no depth model required). Quick cross-slice consistency baseline. |

With no depth, or fewer than 6 clean DA3 views behind it, the processor falls back to a SHARP-only y-ground alignment derived from the median side-slice elevation.

## Depth zones

Each Gaussian is categorised by radial depth from the camera:

| Zone | Range | Treatment |
|------|-------|-----------|
| Align | ≤ 10 m | Kept + used for DA3 scale alignment |
| Near-keep | 10 – 48 m | Kept, not aligned |
| Dead zone | 48 – 50 m | Removed |
| Sky | > 50 m | Kept, not aligned |

## Layout

```
panoramic_to_3dgs/
├── pipeline.py                # Pipeline.run: panorama + depth -> splat
├── config.py                  # PipelineConfig
├── datatype.py                # View
└── components/
    ├── ViewExtractor/         # Equirectangular -> perspective slices
    ├── SplatGenerator/        # Apple SHARP inference
    └── SplatProcessor/        # Alignment, trimming, anchoring
main.py, config.yaml           # Example entry point and settings
```

## Installation

```bash
pip install git+https://github.com/YellowO2/panoramic-to-3dgs.git

# SHARP model
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt -P ./models/
```

## Running

```python
from panoramic_to_3dgs import Pipeline, PipelineConfig

pipeline = Pipeline(PipelineConfig.from_yaml("config.yaml"))
pipeline.run(
    target_appearance_path="pano.jpg",
    output_dir="out",
    depth=depth,   # {"points", "pose", "n_clean"}, e.g. from depth_around
)
```

`target_appearance_path` may be an edited version of the panorama (relit, objects removed): depth is made from the original, SHARP paints the edited one.

## Output

```
output_dir/
├── final_output.ply        # Merged scene, anchored so the target pano center is (0, 0, 0)
└── (debug artifacts when config.debug = true)
```
