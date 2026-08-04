"""SparseDrive <-> NAVSIM coordinate contract (pure numpy).

Contract (NAVSIM_SPARSEDRIVE_PORT.md):

    x_sd = -y_nav          x_nav =  y_sd
    y_sd =  x_nav          y_nav = -x_sd
    yaw_sd = wrap(yaw_nav + pi/2)

Trajectory output: SparseDrive's planner predicts xy only; NAVSIM requires
SE(2). Headings are derived from consecutive positions with an explicit
stationary fallback (displacement threshold 0.05 m; carry forward the last
moving heading; all-stationary trajectories use heading 0 in the current
local frame).
"""
import numpy as np

# NAVSIM-local -> SparseDrive BEV rotation (identical to
# tools/data_converter/navsim_converter.py).
C3 = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]]
)
C4 = np.eye(4)
C4[:3, :3] = C3
YAW_OFFSET = np.pi / 2.0

# Heading is taken from a segment only when it moves at least this far.
STATIONARY_DISP_M = 0.05


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def nav_to_sd_points(pts):
    """Rotate (..., 2 or 3) NAVSIM-local points/vectors into SparseDrive BEV."""
    pts = np.asarray(pts, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = -pts[..., 1]
    out[..., 1] = pts[..., 0]
    return out


def sd_to_nav_points(pts):
    """Inverse rotation: SparseDrive BEV -> NAVSIM-local."""
    pts = np.asarray(pts, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = pts[..., 1]
    out[..., 1] = -pts[..., 0]
    return out


def derive_headings(xy):
    """Headings for a (T, 2) local trajectory (origin at (0, 0)).

    Each heading comes from its preceding segment when the displacement
    exceeds ``STATIONARY_DISP_M``; otherwise the last moving heading is
    carried forward. A fully stationary trajectory gets heading 0
    (= current ego heading in the local frame).
    """
    xy = np.asarray(xy, dtype=np.float64)
    prev = np.zeros(2)
    heading = 0.0
    out = np.zeros(len(xy))
    for i, p in enumerate(xy):
        d = p - prev
        if np.linalg.norm(d) > STATIONARY_DISP_M:
            heading = float(np.arctan2(d[1], d[0]))
        out[i] = heading
        prev = p
    return out


def sd_plan_to_navsim_poses(plan_sd_cum):
    """SparseDrive ``final_planning`` -> NAVSIM SE(2) trajectory poses.

    :param plan_sd_cum: (T, 2) cumulative xy in the SparseDrive BEV frame
        (the model's ``final_planning`` output).
    :return: (T, 3) float32 [x, y, heading] in the NAVSIM local frame
        (rear-axle, x forward / y left), ready for ``navsim ... Trajectory``.
    """
    plan_sd_cum = np.asarray(plan_sd_cum, dtype=np.float64)
    assert plan_sd_cum.ndim == 2 and plan_sd_cum.shape[1] == 2, plan_sd_cum.shape
    xy_nav = sd_to_nav_points(plan_sd_cum)
    headings = derive_headings(xy_nav)
    poses = np.concatenate([xy_nav, headings[:, None]], axis=1)
    return np.asarray(poses, dtype=np.float32)
