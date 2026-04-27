from dataclasses import dataclass
from typing import Optional
import numpy as np
from sharp.utils.gaussians import Gaussians3D

@dataclass
class View:
    path: str # path to the view image
    width: int
    height: int
    yaw: float
    pitch: float
    hfov: float
    vfov: float
    focal_px: float

    # --- Identification and Grouping ---
    pano_id: int = 0  # To group slices from the same panorama

    # --- may or may not have depending on pipeline used ---
    depth: Optional[np.ndarray] = None
    splat: Optional[Gaussians3D] = None
