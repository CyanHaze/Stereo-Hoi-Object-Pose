"""Convert depth_meter.npy (float32, meters) to uint16 PNG (millimeters)."""
import argparse
import numpy as np
import cv2
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="depth_meter.npy file")
    parser.add_argument("--output", required=True, help="output uint16 depth PNG")
    args = parser.parse_args()

    depth_m = np.load(args.input).astype(np.float32)

    # Clamp invalid values to 0
    depth_m[~np.isfinite(depth_m)] = 0
    depth_m[depth_m < 0] = 0

    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, depth_mm)

    valid = depth_m[depth_m > 0]
    print(f"Saved: {args.output}")
    print(f"  shape: {depth_mm.shape}, dtype: {depth_mm.dtype}")
    print(f"  depth meter  min: {float(valid.min()) if len(valid) > 0 else 'N/A':.4f}")
    print(f"  depth meter  max: {float(depth_m.max()):.4f}")
    print(f"  depth mm     max: {depth_mm.max()}")


if __name__ == "__main__":
    main()
