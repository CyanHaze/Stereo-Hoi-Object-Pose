"""Dataset-independent schemas for calibrated multi-view tracking.

Transform names encode their direction. The canonical object state is
``rig_from_object`` and camera extrinsics are stored as ``rig_from_camera``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _matrix(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def _transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = _matrix(value, (4, 4), name)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} must be a homogeneous transform")
    return transform


@dataclass(frozen=True)
class CameraModel:
    """A calibrated camera in a common rig coordinate frame."""

    camera_id: str
    K: np.ndarray
    rig_from_camera: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        object.__setattr__(self, "K", _matrix(self.K, (3, 3), "K"))
        object.__setattr__(
            self,
            "rig_from_camera",
            _transform(self.rig_from_camera, "rig_from_camera"),
        )


@dataclass(frozen=True)
class ViewObservation:
    """Paths and metadata for one synchronized camera observation."""

    frame_id: str
    camera: CameraModel
    rgb_path: Path | None = None
    depth_path: Path | None = None
    mask_path: Path | None = None
    visibility: float | None = None
    valid: bool = True
    source: str = "dataset"

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        for name in ("rgb_path", "depth_path", "mask_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if self.visibility is not None and not 0.0 <= self.visibility <= 1.0:
            raise ValueError("visibility must be in [0, 1]")


@dataclass(frozen=True)
class FrameBundle:
    """All synchronized observations for one frame."""

    frame_id: str
    views: tuple[ViewObservation, ...]
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("FrameBundle requires at least one view")
        if any(view.frame_id != self.frame_id for view in self.views):
            raise ValueError("all views must have the FrameBundle frame_id")
        camera_ids = [view.camera.camera_id for view in self.views]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera_id values must be unique within a frame")


@dataclass(frozen=True)
class PoseEstimate:
    """One estimated object state and its reliability metadata."""

    frame_id: str
    rig_from_object: np.ndarray
    confidence: float | None = None
    covariance: np.ndarray | None = None
    active_views: tuple[str, ...] = ()
    rejected_views: tuple[str, ...] = ()
    recovered: bool = False
    runtime_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rig_from_object",
            _transform(self.rig_from_object, "rig_from_object"),
        )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.covariance is not None:
            object.__setattr__(
                self,
                "covariance",
                _matrix(self.covariance, (6, 6), "covariance"),
            )
        if self.runtime_ms is not None and self.runtime_ms < 0:
            raise ValueError("runtime_ms must be non-negative")

