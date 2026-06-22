"""Pose-sequence input helpers."""

from pathlib import Path

import numpy as np

from ..geometry.se3 import validate_transform


def load_pose_directory(directory: str | Path) -> dict[str, np.ndarray]:
    """Load ``<frame_id>.txt`` homogeneous transforms from a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"pose directory not found: {directory}")
    poses: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("*.txt")):
        matrix = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
        poses[path.stem] = validate_transform(matrix, str(path))
    if not poses:
        raise FileNotFoundError(f"no .txt pose files in {directory}")
    return poses


def align_pose_directories(predicted_dir: str | Path,
                           ground_truth_dir: str | Path,
                           ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load and align predicted and ground-truth poses by frame identifier."""
    predicted = load_pose_directory(predicted_dir)
    ground_truth = load_pose_directory(ground_truth_dir)
    frame_ids = sorted(set(predicted) & set(ground_truth))
    if not frame_ids:
        raise ValueError("predicted and ground-truth directories share no frame IDs")
    pred_array = np.stack([predicted[frame_id] for frame_id in frame_ids])
    gt_array = np.stack([ground_truth[frame_id] for frame_id in frame_ids])
    return frame_ids, pred_array, gt_array

