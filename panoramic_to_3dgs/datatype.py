from dataclasses import dataclass
from typing import Optional

from sharp.utils.gaussians import Gaussians3D


@dataclass
class View:
    """One perspective slice of a panorama, and the splat SHARP made of it."""
    path: str
    width: int
    height: int
    yaw: float
    pitch: float
    hfov: float
    vfov: float
    focal_px: float
    pano_id: int | str = 0
    splat: Optional[Gaussians3D] = None
