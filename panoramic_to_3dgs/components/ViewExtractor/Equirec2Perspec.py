# from https://github.com/fuenwang/Equirec2Perspec
import cv2
import numpy as np

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out

def lonlat2xyz(lonlat):
    lon = lonlat[..., 0:1]
    lat = lonlat[..., 1:2]
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.concatenate([x, y, z], axis=-1)


def rotate_equirectangular(img, heading: float, roll: float, pitch: float):
    """Rotate an equirectangular image about its own center: `pitch` (nose
    up/down) and `roll` (side to side), both relative to the vehicle's own
    direction of travel (`heading`), all in radians. Casts each output pixel
    to a sphere direction, rotates it, casts back to a source pixel, and
    remaps.

    `heading` matters because the image's own lon=0 axis is north-referenced,
    not the vehicle's forward direction — pitch/roll must be applied about
    axes rotated by heading, not the image's fixed lon=0/lon=90 axes.

    Used to correct for a panorama's own upright-correction tilt (e.g. from
    Street View's pitch/roll metadata) before slicing it into perspective
    views, so slices show a level-looking ground instead of one tilted by the
    local road grade.
    """
    height_px, width = img.shape[:2]
    x = np.arange(width)
    y = np.arange(height_px)
    x, y = np.meshgrid(x, y)
    XY = np.stack([x, y], axis=-1).astype(np.float32)
    lonlat_out = np.stack(
        [
            (XY[..., 0] / (width - 1) - 0.5) * (2 * np.pi),
            (XY[..., 1] / (height_px - 1) - 0.5) * np.pi,
        ],
        axis=-1,
    )
    xyz_out = lonlat2xyz(lonlat_out)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    R_heading, _ = cv2.Rodrigues(y_axis * heading)
    forward_local = R_heading @ np.array([0.0, 0.0, 1.0], np.float32)
    right_local = R_heading @ np.array([1.0, 0.0, 0.0], np.float32)

    R_pitch, _ = cv2.Rodrigues(right_local * pitch)
    R_roll, _ = cv2.Rodrigues(forward_local * roll)
    R = R_roll @ R_pitch

    xyz_src = xyz_out @ R
    lonlat_src = xyz2lonlat(xyz_src)
    XY_src = lonlat2XY(lonlat_src, shape=img.shape).astype(np.float32)
    return cv2.remap(img, XY_src[..., 0], XY_src[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


class Equirectangular:
    def __init__(self, img_data):
        if isinstance(img_data, str):
            self._img = cv2.imread(img_data, cv2.IMREAD_COLOR)
        else:
            self._img = img_data
            
        if len(self._img.shape) == 2:
            self._height, self._width = self._img.shape
        else:
            self._height, self._width, _ = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz) 
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp