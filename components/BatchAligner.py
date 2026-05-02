import numpy as np


def compute_local_to_world(
    R_local: np.ndarray, C_local: np.ndarray,
    R_world: np.ndarray, C_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the same physical camera's pose expressed in two frames, compute the
    rigid transform that maps the local frame into the world frame.

    R_local / R_world : camera-to-world rotation (3×3) in each frame
    C_local / C_world : camera centre (3,) in each frame

    Returns R_l2w (3×3), t_l2w (3,) such that for any camera k in the batch:
        C_k_world = R_l2w @ C_k_local + t_l2w
        R_k_world = R_l2w @ R_k_local
    """
    R_l2w = R_world.T @ R_local
    t_l2w = C_world - R_l2w @ C_local
    return R_l2w, t_l2w


def apply_to_pose(
    pose: dict, R_l2w: np.ndarray, t_l2w: np.ndarray
) -> dict:
    """Transform a {'center': ..., 'rotation': ...} pose dict from local to world frame.

    pose['rotation'] is R_w2p (world-to-pano).  Under frame change F_local→F_world:
        R_w2p_world = R_w2p_local @ R_l2w.T
        C_world     = R_l2w @ C_local + t_l2w
    """
    return {
        'center': R_l2w @ pose['center'] + t_l2w,
        'rotation': pose['rotation'] @ R_l2w.T,
    }


def apply_to_pts(
    pts: np.ndarray, R_l2w: np.ndarray, t_l2w: np.ndarray
) -> np.ndarray:
    """Transform an (N, 3) point array from local to world frame."""
    return (R_l2w @ pts.T).T + t_l2w
