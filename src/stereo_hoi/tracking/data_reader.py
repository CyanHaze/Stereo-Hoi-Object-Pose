"""Data reader for FoundationPose tracking — left or right camera view.

Right-camera depth and mask are automatically warped from the left-camera
FFS depth using the stereo baseline.  No separate FFS run is needed for the
right view.
"""

import glob
import json
import logging
import os

import cv2
import numpy as np

from ..depth.warp import warp_left_depth_to_right, warp_mask_to_right


class FPDataReader:
    """Reads RGB, FFS depth, mask, and K for FoundationPose tracking.

    Parameters
    ----------
    data_dir:
        Path to the clip directory (``data/<clip>/``).
    camera:
        ``'left'`` or ``'right'``.
    shorter_side:
        If set, downscale images so the shorter side equals this value.
    zfar:
        Far-plane depth (metres).  Pixels beyond this are treated as invalid.
    """

    def __init__(self, data_dir: str, camera: str = "left",
                 shorter_side: int | None = None, zfar: float = np.inf):
        self.data_dir = data_dir
        self.camera = camera
        self.zfar = zfar

        # ---- RGB ----
        rgb_dir = "rgb" if camera == "left" else "right"
        pattern = os.path.join(data_dir, rgb_dir, "*.jpg")
        self.color_files = sorted(glob.glob(pattern))
        if not self.color_files:
            self.color_files = sorted(glob.glob(os.path.join(data_dir, rgb_dir, "*.png")))
        if not self.color_files:
            raise FileNotFoundError(f"No images in {data_dir}/{rgb_dir}/")

        self.id_strs = [os.path.splitext(os.path.basename(f))[0]
                        for f in self.color_files]

        # ---- intrinsics ----
        K_path = os.path.join(data_dir, "ffs", "cam_K.txt")
        self.K = np.loadtxt(K_path).reshape(3, 3)

        # ---- baseline (for right-camera warp) ----
        calib_path = os.path.join(data_dir, "calib.json")
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                self.baseline_m = float(json.load(f)["baseline_m"])
        else:
            self.baseline_m = 0.0

        # ---- resolution ----
        H, W = cv2.imread(self.color_files[0]).shape[:2]
        self.H_orig, self.W_orig = H, W
        if shorter_side is not None:
            downscale = shorter_side / min(H, W)
            self.H = int(H * downscale)
            self.W = int(W * downscale)
            self.K = self.K.copy()
            self.K[:2] *= downscale
        else:
            self.H, self.W = H, W

        logging.info(
            "[%s] %dx%d, %d frames, baseline=%.4f m",
            camera, self.W, self.H, len(self), self.baseline_m,
        )

    def __len__(self) -> int:
        return len(self.color_files)

    def get_color(self, i: int) -> np.ndarray:
        """Read and resize RGB for frame *i*."""
        import imageio
        color = imageio.imread(self.color_files[i])
        if color.ndim == 2:
            color = np.tile(color[..., None], (1, 1, 3))
        color = color[..., :3]
        return cv2.resize(color, (self.W, self.H),
                          interpolation=cv2.INTER_NEAREST)

    def get_depth(self, i: int) -> np.ndarray:
        """Read left FFS depth (uint16 mm → metres), warp if right camera."""
        depth_path = os.path.join(self.data_dir, "ffs", "depth",
                                  f"{self.id_strs[i]}.png")
        depth_mm = cv2.imread(depth_path, -1).astype(np.float32)
        depth_m = depth_mm / 1000.0
        depth_m = cv2.resize(depth_m, (self.W, self.H),
                             interpolation=cv2.INTER_NEAREST)

        if self.camera == "right" and self.baseline_m > 0:
            depth_m = warp_left_depth_to_right(depth_m, self.K, self.baseline_m)

        depth_m[(depth_m < 0.001) | (depth_m >= self.zfar)] = 0
        return depth_m

    def get_mask(self, i: int) -> np.ndarray:
        """Read object mask, warp to right camera if needed."""
        mask_path = os.path.join(self.data_dir, "mask", "object",
                                 f"{self.id_strs[i]}.png")
        if not os.path.exists(mask_path):
            return np.zeros((self.H, self.W), dtype=np.uint8)

        mask = cv2.imread(mask_path, -1)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = cv2.resize(mask, (self.W, self.H),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        if self.camera == "right" and self.baseline_m > 0:
            depth_path = os.path.join(self.data_dir, "ffs", "depth",
                                      f"{self.id_strs[i]}.png")
            depth_left_mm = cv2.imread(depth_path, -1).astype(np.float32)
            depth_left = depth_left_mm / 1000.0
            depth_left = cv2.resize(depth_left, (self.W, self.H),
                                    interpolation=cv2.INTER_NEAREST)
            mask = warp_mask_to_right(mask, self.K, self.baseline_m,
                                      depth_left)

        return mask
