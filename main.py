"""Make a splat from one panorama.

Depth comes from outside this package: pass what
streetview_to_3d's da3_ops.depth_around returns as `depth`. Without it the
slices are aligned to each other only.
"""
from panoramic_to_3dgs import Pipeline, PipelineConfig

if __name__ == "__main__":
    pipeline = Pipeline(PipelineConfig.from_yaml("config.yaml"))
    pipeline.run(
        target_appearance_path="data/inputs/panoramas_sea_view/pano_rTCgvONHkRFIqvygt6llLA.jpg",
        output_dir="data/outputs/sea_view",
        depth=None,
    )
