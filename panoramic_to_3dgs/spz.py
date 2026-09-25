"""Write a splat as SPZ, Niantic's compressed splat format (version 2).

About 19 bytes a splat before gzip, against the PLY's 56: positions as
24-bit fixed point, everything else as one byte per value. Follows the
reference encoder (nianticlabs/spz, load-spz.cc packGaussians) field by
field, and version 2 because it is the newest one Spark 0.1.10 -- the
viewer -- reads without the v3 quaternion packing.

Coordinates are written as they are, not converted to SPZ's nominal RUB
axes: Spark reads SPZ and PLY positions alike, raw, so the viewer's own
flip keeps working unchanged.
"""
import gzip
import math
import struct

import numpy as np

MAGIC = 0x5053474E  # "NGSP"
VERSION = 2
COLOR_SCALE = 0.15  # the reference's colorScale
MAX_FRACTIONAL_BITS = 12  # the reference's ~0.25 mm


def _u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def encode_spz(xyz, opacity, sh_dc, scale, quat_wxyz) -> bytes:
    """The gzipped SPZ bytes for N splats. opacity in [0, 1], scale in
    metres (not log), sh_dc the degree-0 SH colour, quat_wxyz scalar-first."""
    xyz = np.asarray(xyz, np.float64)
    n = len(xyz)

    # 24-bit signed fixed point. Sky splats can sit kilometres out, past the
    # reference's fixed 12 fractional bits (+-2 km), so fewer bits are used
    # when the farthest splat needs them -- the header records how many.
    far = float(np.abs(xyz).max()) if n else 0.0
    bits = MAX_FRACTIONAL_BITS if far < 1 else max(0, min(MAX_FRACTIONAL_BITS, 22 - math.ceil(math.log2(far))))
    fixed = np.clip(np.round(xyz * (1 << bits)), -(1 << 23), (1 << 23) - 1).astype(np.int32).reshape(-1)
    positions = np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], 1).astype(np.uint8)

    alphas = _u8(np.asarray(opacity, np.float64).reshape(-1) * 255)
    colors = _u8(np.asarray(sh_dc, np.float64) * (COLOR_SCALE * 255) + 0.5 * 255)
    scales = _u8((np.log(np.asarray(scale, np.float64)) + 10) * 16)

    # SPZ v2 keeps x, y, z of the normalised quaternion with w made
    # non-negative; the reader derives w
    q = np.asarray(quat_wxyz, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)
    rotations = _u8(q[:, 1:] * 127.5 + 127.5)

    raw = struct.pack("<IIIBBBB", MAGIC, VERSION, n, 0, bits, 0, 0) + b"".join(
        np.ascontiguousarray(a).tobytes() for a in (positions, alphas, colors, scales, rotations))
    return gzip.compress(raw, compresslevel=6)


def save_spz(gaussians, path: str) -> None:
    """Save a sharp Gaussians3D, with colours converted exactly as sharp's
    own save_ply does (linear RGB -> sRGB -> SH DC)."""
    from sharp.utils import color_space as cs_utils
    from sharp.utils.gaussians import convert_rgb_to_spherical_harmonics

    def flat(t):
        return t.flatten(0, 1).detach().cpu().double().numpy()

    dc = convert_rgb_to_spherical_harmonics(cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1)))
    with open(path, "wb") as f:
        f.write(encode_spz(flat(gaussians.mean_vectors), flat(gaussians.opacities),
                           dc.detach().cpu().double().numpy(), flat(gaussians.singular_values),
                           flat(gaussians.quaternions)))
