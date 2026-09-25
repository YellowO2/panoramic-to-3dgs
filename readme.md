# Panoramic to 3DGS

Turn one equirectangular panorama into a 3D Gaussian Splat. Try it in the **Panorama → 3DGS** tab of the [Hugging Face Space](https://huggingface.co/spaces/potato-bug/street-view-to-3d).

<table>
<tr>
<td align="center"><sub>Demo Video</sub></td>
<td align="center"><sub>Comparison with HunyuanWorld 2.0 + World Marble 1.1</sub></td>
</tr>
<tr>
<td width="50%"><a href="https://youtu.be/mzIDZWxv4vA"><img src="https://img.youtube.com/vi/mzIDZWxv4vA/hqdefault.jpg" alt="Demo video"></a></td>
<td width="50%"><a href="https://youtu.be/fYANbQXMZ_0"><img src="https://img.youtube.com/vi/fYANbQXMZ_0/maxresdefault.jpg" alt="Comparison with HunyuanWorld 2.0 + World Marble 1.1"></a></td>
</tr>
</table>

Not a standalone tool: the splat is scaled against depth made elsewhere, normally by [streetview-to-3d](https://github.com/YellowO2/streetview-to-3d) from the panorama and its neighbours. Without depth it still runs, but only aligns the slices to each other.

## Scale modes

| Mode | Description |
|------|-------------|
| `da3_2dgrid_global` *(recommended)* | Scale each Gaussian by the median DA3/SHARP ratio inside a 2D (depth × FOV) cell, then smooth across cells. |
| `da3_y_ground` | Uniform per-slice scale so SHARP's ground elevation matches DA3's ground elevation. Simpler and more robust when depth coverage is patchy. |
| `near_edge` | Match nearest-Z across slices (no depth model required). Quick cross-slice consistency baseline. |

With no depth, or fewer than 6 clean DA3 views behind it, the processor falls back to a SHARP-only y-ground alignment derived from the median side-slice elevation.

## Usage

```bash
pip install git+https://github.com/YellowO2/panoramic-to-3dgs.git
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt -P ./models/
```

```python
from panoramic_to_3dgs import Pipeline, PipelineConfig

pipeline = Pipeline(PipelineConfig.from_yaml("config.yaml"))
pipeline.run("pano.jpg", output_dir="out", depth=depth)  # writes out/final_output.ply
```

`depth` is `{"points", "pose", "n_clean"}`, as streetview-to-3d's `da3_ops.depth_around` returns it, or `None`.

## Acknowledgments

- [Apple ml-sharp](https://github.com/apple/ml-sharp) (Apple sample code license)
- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache 2.0), for the depth it is aligned to
