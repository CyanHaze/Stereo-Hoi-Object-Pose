"""Small NumPy SE(3) utilities with explicit transform directions."""

import numpy as np


def validate_transform(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    """Return a validated copy of a homogeneous 4x4 transform."""
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} is not a homogeneous transform")
    return value.copy()


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    value = validate_transform(transform)
    rotation = value[:3, :3]
    translation = value[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def relative_transform(reference_from_object_a: np.ndarray,
                       reference_from_object_b: np.ndarray) -> np.ndarray:
    """Return ``object_a_from_object_b`` from two poses in one reference frame."""
    return invert_transform(reference_from_object_a) @ validate_transform(
        reference_from_object_b
    )


def camera_from_object(rig_from_camera: np.ndarray,
                       rig_from_object: np.ndarray) -> np.ndarray:
    """Convert the canonical rig-frame object state to one camera frame."""
    return invert_transform(rig_from_camera) @ validate_transform(rig_from_object)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` point array."""
    value = validate_transform(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    return (value[:3, :3] @ points.T).T + value[:3, 3]


def rotation_geodesic_deg(rotation_a: np.ndarray,
                          rotation_b: np.ndarray) -> float:
    """Geodesic angular distance between two 3x3 rotations, in degrees."""
    rotation_a = np.asarray(rotation_a, dtype=np.float64)
    rotation_b = np.asarray(rotation_b, dtype=np.float64)
    if rotation_a.shape != (3, 3) or rotation_b.shape != (3, 3):
        raise ValueError("rotation matrices must have shape (3, 3)")
    delta = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

