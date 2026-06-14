"""Pure-numpy rotation utilities + multi-view pose fusion strategies."""

import numpy as np


# ---------------------------------------------------------------------------
# Rotation (drop-in for scipy.spatial.transform.Rotation)
# ---------------------------------------------------------------------------

class Rotation:
    """Minimal pure-numpy 3D rotation represented as a unit quaternion."""

    def __init__(self, quat):
        self._q = np.asarray(quat, dtype=np.float64)

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_matrix(cls, mat):
        """3x3 rotation matrix → Rotation."""
        m = np.asarray(mat, dtype=np.float64)
        trace = np.trace(m)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return cls([x, y, z, w])

    @classmethod
    def from_quat(cls, q):
        """Quaternion [x, y, z, w] → Rotation."""
        return cls(q)

    # -- converters ----------------------------------------------------------

    def as_quat(self):
        """Return [x, y, z, w]."""
        return self._q.copy()

    def as_matrix(self):
        """Return 3x3 rotation matrix."""
        x, y, z, w = self._q
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ], dtype=np.float64)


R = Rotation  # shorthand alias


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def build_left_from_right(baseline_m: float) -> np.ndarray:
    """Right camera → left camera transform for rectified stereo.

    The right camera is at [baseline, 0, 0] in the left-camera frame.
    Returns a 4×4 homogeneous matrix: ``T_left_from_right``.
    """
    T = np.eye(4)
    T[0, 3] = baseline_m
    return T


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_poses(txt_dir: str, id_strs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load 4×4 pose matrices from a directory of .txt files.

    Returns ``(poses, missing)`` where *poses* is ``(N, 4, 4)`` and
    *missing* is a boolean array of length *N*.
    """
    N = len(id_strs)
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    missing = np.zeros(N, dtype=bool)
    for i, sid in enumerate(id_strs):
        path = os.path.join(txt_dir, f"{sid}.txt")
        if os.path.exists(path):
            poses[i] = np.loadtxt(path).reshape(4, 4)
        else:
            missing[i] = True
    return poses, missing


def save_poses(txt_dir: str, id_strs: list[str], poses: np.ndarray) -> None:
    """Save 4×4 poses as ``<id>.txt`` files."""
    os.makedirs(txt_dir, exist_ok=True)
    for i, sid in enumerate(id_strs):
        np.savetxt(os.path.join(txt_dir, f"{sid}.txt"), poses[i].reshape(4, 4))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_available(poses_left_i: np.ndarray, poses_right_i: np.ndarray) -> np.ndarray:
    """Return whichever of left/right has a non-zero pose."""
    if not np.all(poses_left_i == 0):
        return poses_left_i.copy()
    return poses_right_i.copy()


def _rotation_diff_deg(R_l: np.ndarray, R_r: np.ndarray) -> float:
    """Angular distance between two 3×3 rotation matrices, in degrees."""
    dR = R_l.T @ R_r
    cos = (np.trace(dR) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _mean_rotation(r_l: Rotation, r_r: Rotation) -> Rotation:
    """Quaternion average with sign-disambiguation."""
    q_l = r_l.as_quat()
    q_r = r_r.as_quat()
    if np.dot(q_l, q_r) < 0:
        q_r = -q_r
    q = q_l + q_r
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q)


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

def fuse_average(poses_left: np.ndarray, poses_right: np.ndarray,
                 valid: np.ndarray) -> np.ndarray:
    """Equal-weight fusion: quaternion-mean rotation, arithmetic-mean translation.

    Args:
        poses_left:  ``(N, 4, 4)`` — left-camera poses.
        poses_right: ``(N, 4, 4)`` — right-camera poses (already in left frame).
        valid:       ``(N,)`` bool — frames where both cameras have a result.
    """
    N = len(valid)
    fused = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        if not valid[i]:
            fused[i] = _copy_available(poses_left[i], poses_right[i])
            continue

        r_l = Rotation.from_matrix(poses_left[i, :3, :3])
        r_r = Rotation.from_matrix(poses_right[i, :3, :3])
        r_fused = _mean_rotation(r_l, r_r)
        t_fused = (poses_left[i, :3, 3] + poses_right[i, :3, 3]) / 2.0

        fused[i, :3, :3] = r_fused.as_matrix()
        fused[i, :3, 3] = t_fused
        fused[i, 3, 3] = 1.0
    return fused


def fuse_left_main(poses_left: np.ndarray, poses_right: np.ndarray,
                   valid: np.ndarray, trans_thresh: float = 0.02,
                   rot_thresh_deg: float = 10.0) -> np.ndarray:
    """Left-main fusion: use left by default, average only when L/R agree.

    If translation diff < *trans_thresh* (m) AND rotation diff
    < *rot_thresh_deg*, average; otherwise keep left.
    """
    N = len(valid)
    fused = np.zeros((N, 4, 4), dtype=np.float64)
    n_avg, n_left = 0, 0
    for i in range(N):
        if not valid[i]:
            fused[i] = _copy_available(poses_left[i], poses_right[i])
            continue

        t_l = poses_left[i, :3, 3]
        t_r = poses_right[i, :3, 3]
        trans_diff = float(np.linalg.norm(t_l - t_r))
        rot_diff = _rotation_diff_deg(poses_left[i, :3, :3],
                                      poses_right[i, :3, :3])

        if trans_diff < trans_thresh and rot_diff < rot_thresh_deg:
            r_l = Rotation.from_matrix(poses_left[i, :3, :3])
            r_r = Rotation.from_matrix(poses_right[i, :3, :3])
            r_fused = _mean_rotation(r_l, r_r)
            t_fused = (t_l + t_r) / 2.0
            fused[i, :3, :3] = r_fused.as_matrix()
            fused[i, :3, 3] = t_fused
            n_avg += 1
        else:
            fused[i] = poses_left[i].copy()
            n_left += 1

        fused[i, 3, 3] = 1.0

    import logging
    logging.info(
        "  left_main: %d averaged, %d left-only "
        "(thresholds: trans<%.3fm, rot<%.0fdeg)",
        n_avg, n_left, trans_thresh, rot_thresh_deg,
    )
    return fused


# ---------------------------------------------------------------------------
import os  # noqa: E402 (used by load_poses / save_poses above)
