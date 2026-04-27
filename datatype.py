from dataclasses import dataclass
from typing import Optional
from cv2.typing import MatLike
import numpy as np
from sharp.utils.gaussians import Gaussians3D

@dataclass
class View:
    path: str
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
    image: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    splat: Optional[Gaussians3D] = None
    extrinsics: Optional[np.ndarray] = None  # 3x4 W2C matrix, if using DA3 (or pose info)