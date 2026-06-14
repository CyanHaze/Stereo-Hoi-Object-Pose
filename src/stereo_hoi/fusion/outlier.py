"""Outlier detection: flag frames where left/right tracking disagree."""

import logging
import numpy as np
from .core import _rotation_diff_deg


def detect_outliers(poses_left: np.ndarray, poses_right: np.ndarray,
                    valid: np.ndarray, trans_thresh: float = 0.05,
                    rot_thresh_deg: float = 30.0) -> np.ndarray:
    """Mark frames where right tracking likely failed as invalid.

    A frame is an outlier if the left–right translation difference exceeds
    *trans_thresh* (metres) OR the left–right rotation difference exceeds
    *rot_thresh_deg*.

    Returns an updated *valid* boolean array ``(N,)``.
    """
    valid_out = valid.copy()
    n_outliers = 0
    diffs = []
    for i in range(len(valid)):
        if not valid[i]:
            continue
        t_diff = float(np.linalg.norm(
            poses_left[i, :3, 3] - poses_right[i, :3, 3]))
        r_diff = _rotation_diff_deg(
            poses_left[i, :3, :3], poses_right[i, :3, :3])
        diffs.append((t_diff, r_diff))
        if t_diff > trans_thresh or r_diff > rot_thresh_deg:
            valid_out[i] = False
            n_outliers += 1

    diffs = np.array(diffs) if diffs else np.zeros((0, 2))
    if len(diffs) > 0:
        logging.info(
            "  Outlier check: %d/%d frames rejected "
            "(trans>%.0fmm | rot>%.0fdeg)",
            n_outliers, len(diffs), trans_thresh * 1000, rot_thresh_deg,
        )
        logging.info(
            "  L-R stats (valid frames): "
            "trans median=%.1fmm max=%.1fmm | "
            "rot median=%.1fdeg max=%.1fdeg",
            float(np.median(diffs[:, 0])) * 1000,
            float(diffs[:, 0].max()) * 1000,
            float(np.median(diffs[:, 1])),
            float(diffs[:, 1].max()),
        )
    return valid_out
