"""Temporal pose smoothing (quaternion + translation space)."""

import numpy as np
from .core import Rotation


def _gaussian_kernel(size: int, sigma: float | None = None) -> np.ndarray:
    """1D Gaussian kernel, normalised to sum to 1."""
    if sigma is None:
        sigma = size / 4.0
    x = np.arange(size) - (size - 1) / 2.0
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_poses(poses: np.ndarray, window: int = 5,
                 method: str = "gaussian") -> np.ndarray:
    """Smooth a pose sequence in quaternion + translation space.

    Args:
        poses:  ``(N, 4, 4)`` pose array.
        window: kernel size (odd integer).
        method: ``'gaussian'`` (default) or ``'moving_avg'``.
    """
    if window <= 1:
        return poses.copy()

    if window % 2 == 0:
        window += 1

    N = poses.shape[0]
    quats = np.zeros((N, 4))
    trans = np.zeros((N, 3))
    for i in range(N):
        quats[i] = Rotation.from_matrix(poses[i, :3, :3]).as_quat()
        trans[i] = poses[i, :3, 3]

    # Sign-disambiguate consecutive quaternions
    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    kernel = _gaussian_kernel(window) if method == "gaussian" \
        else np.ones(window) / window

    half = window // 2
    smoothed_quat = np.zeros_like(quats)
    smoothed_trans = np.zeros_like(trans)

    for i in range(N):
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        k_lo = half - (i - lo)
        k_hi = half + (hi - i)
        k = kernel[k_lo:k_hi]
        k = k / k.sum()
        smoothed_quat[i] = (quats[lo:hi].T @ k).T
        smoothed_trans[i] = (trans[lo:hi].T @ k).T

    norms = np.linalg.norm(smoothed_quat, axis=1, keepdims=True)
    smoothed_quat /= norms

    smoothed = poses.copy()
    for i in range(N):
        smoothed[i, :3, :3] = Rotation.from_quat(smoothed_quat[i]).as_matrix()
        smoothed[i, :3, 3] = smoothed_trans[i]
    return smoothed
