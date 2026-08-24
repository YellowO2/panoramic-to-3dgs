from dataclasses import dataclass
from typing import Optional
from sharp.utils.gaussians import Gaussians3D
from panoramic_da3 import View as DA3View


@dataclass
class View(DA3View):
    # SHARP's own generated splat for this view -- not something
    # panoramic_da3's View knows about, since that package has no SHARP/GS
    # step at all.
    splat: Optional[Gaussians3D] = None
