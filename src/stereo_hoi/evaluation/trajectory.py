"""Sequence-level metrics for accuracy, drift, failure, and recovery."""

import numpy as np

from ..geometry.se3 import invert_transform
from .pose import pose_errors, pose_success_mask


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _failure_runs(failed: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(failed)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def relative_pose_errors(predicted: np.ndarray,
                         ground_truth: np.ndarray,
                         delta: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Relative pose error over a fixed frame offset."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    if len(predicted) != len(ground_truth):
        raise ValueError("predicted and ground_truth must have equal length")
    if len(predicted) <= delta:
        return np.zeros(0), np.zeros(0)

    relative_pred = []
    relative_gt = []
    for index in range(len(predicted) - delta):
        relative_pred.append(invert_transform(predicted[index]) @ predicted[index + delta])
        relative_gt.append(
            invert_transform(ground_truth[index]) @ ground_truth[index + delta]
        )
    return pose_errors(np.stack(relative_pred), np.stack(relative_gt))


def evaluate_trajectory(predicted: np.ndarray,
                        ground_truth: np.ndarray,
                        *,
                        relative_delta: int = 1,
                        failure_translation_m: float = 0.05,
                        failure_rotation_deg: float = 30.0,
                        ) -> dict:
    """Compute a JSON-serializable sequence evaluation report."""
    translation, rotation = pose_errors(predicted, ground_truth)
    success = pose_success_mask(
        translation,
        rotation,
        translation_threshold_m=failure_translation_m,
        rotation_threshold_deg=failure_rotation_deg,
    )
    failed = ~success
    runs = _failure_runs(failed)
    relative_translation, relative_rotation = relative_pose_errors(
        predicted, ground_truth, delta=relative_delta
    )

    return {
        "num_frames": int(len(predicted)),
        "thresholds": {
            "failure_translation_m": float(failure_translation_m),
            "failure_rotation_deg": float(failure_rotation_deg),
            "relative_delta": int(relative_delta),
        },
        "absolute_translation_m": _summary(translation),
        "absolute_rotation_deg": _summary(rotation),
        "relative_translation_m": _summary(relative_translation),
        "relative_rotation_deg": _summary(relative_rotation),
        "success_rate": float(success.mean()),
        "failure_rate": float(failed.mean()),
        "failure_episodes": int(len(runs)),
        "longest_failure_run": int(max((end - start + 1 for start, end in runs),
                                       default=0)),
        "time_to_first_failure_frames": (
            int(np.flatnonzero(failed)[0]) if failed.any() else None
        ),
        "failure_runs": [
            {"start_index": start, "end_index": end, "length": end - start + 1}
            for start, end in runs
        ],
    }

