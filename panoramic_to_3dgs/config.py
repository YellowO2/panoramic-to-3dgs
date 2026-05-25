from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class PipelineConfig:
    # Model paths
    sharp_model: str = ""
    da3_model: str = ""
    da360_model: str = ""

    # Pipeline
    depth_mode: Optional[str] = "da3"  # 'da3' | 'da360' | None
    scale_mode: str = "da3_2dgrid_global"
    clean_image: bool = False
    slice_count: int = 6
    include_sky: bool = False  # include an upward (+90° pitch) SHARP view
    debug: bool = False  # save intermediate view slices and debug PCDs

    # SplatProcessor
    align_depth: float = 10.0
    near_depth: float = 48.0
    sky_depth: float = 50.0
    num_z_slabs: int = 500
    num_fov_slabs: int = 250
    smooth_sigma_m: float = 0.5
    smooth_sigma_fov: float = 0.15
    voronoi_buffer_m: float = 1.5
    floor_keep_fraction: float = 0.6
    min_depth_coverage: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        models = data.get("models", {})
        pipeline = data.get("pipeline", {})
        processor = data.get("processor", {})
        return cls(
            sharp_model=models.get("sharp", ""),
            da3_model=models.get("da3", ""),
            da360_model=models.get("da360", ""),
            depth_mode=pipeline.get("depth_mode", "da3"),
            scale_mode=pipeline.get("scale_mode", "da3_2dgrid_global"),
            clean_image=pipeline.get("clean_image", False),
            slice_count=pipeline.get("slice_count", 6),
            include_sky=pipeline.get("include_sky", False),
            debug=pipeline.get("debug", False),
            align_depth=processor.get("align_depth", 10.0),
            near_depth=processor.get("near_depth", 48.0),
            sky_depth=processor.get("sky_depth", 50.0),
            num_z_slabs=processor.get("num_z_slabs", 500),
            num_fov_slabs=processor.get("num_fov_slabs", 250),
            smooth_sigma_m=processor.get("smooth_sigma_m", 0.5),
            smooth_sigma_fov=processor.get("smooth_sigma_fov", 0.15),
            voronoi_buffer_m=processor.get("voronoi_buffer_m", 1.5),
            floor_keep_fraction=processor.get("floor_keep_fraction", 0.6),
            min_depth_coverage=processor.get("min_depth_coverage", 1.0),
        )
