"""Frame-level 6D pose metrics."""

import numpy as np

from ..geometry.se3 import transform_points


def _pose_array(poses: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(poses, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (N, 4, 4), got {value.shape}")
    return value


def pose_errors(predicted: np.ndarray,
                ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return translation errors in meters and rotation errors in degrees."""
    predicted = _pose_array(predicted, "predicted")
    ground_truth = _pose_array(ground_truth, "ground_truth")
    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground_truth must have the same shape")

    translation = np.linalg.norm(
        predicted[:, :3, 3] - ground_truth[:, :3, 3], axis=1
    )
    delta = np.swapaxes(predicted[:, :3, :3], 1, 2) @ ground_truth[:, :3, :3]
    cosine = np.clip(
        (np.trace(delta, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
    )
    rotation = np.degrees(np.arccos(cosine))
    return translation, rotation


def pose_success_mask(translation_errors_m: np.ndarray,
                      rotation_errors_deg: np.ndarray,
                      translation_threshold_m: float = 0.05,
                      rotation_threshold_deg: float = 30.0) -> np.ndarray:
    """Threshold frame-level pose errors using a joint success criterion."""
    translation_errors_m = np.asarray(translation_errors_m, dtype=np.float64)
    rotation_errors_deg = np.asarray(rotation_errors_deg, dtype=np.float64)
    if translation_errors_m.shape != rotation_errors_deg.shape:
        raise ValueError("translation and rotation errors must have the same shape")
    return ((translation_errors_m <= translation_threshold_m)
            & (rotation_errors_deg <= rotation_threshold_deg))


def add_errors(model_points: np.ndarray,
               predicted: np.ndarray,
               ground_truth: np.ndarray) -> np.ndarray:
    """Average Distance of Model Points for asymmetric objects."""
    model_points = np.asarray(model_points, dtype=np.float64)
    if model_points.ndim != 2 or model_points.shape[1] != 3:
        raise ValueError("model_points must have shape (M, 3)")
    predicted = _pose_array(predicted, "predicted")
    ground_truth = _pose_array(ground_truth, "ground_truth")
    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground_truth must have the same shape")

    errors = np.empty(len(predicted), dtype=np.float64)
    for index, (pred_pose, gt_pose) in enumerate(zip(predicted, ground_truth)):
        pred_points = transform_points(pred_pose, model_points)
        gt_points = transform_points(gt_pose, model_points)
        errors[index] = np.linalg.norm(pred_points - gt_points, axis=1).mean()
    return errors

