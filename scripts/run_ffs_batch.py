#!/usr/bin/env python
"""
Batch FFS (Fast-FoundationStereo) depth generation for a clip.

Processes all stereo pairs in range, saves uint16 mm depth PNGs directly
to data/{clip}/ffs/depth/ — ready to be consumed by FoundationPose.

Usage:
    conda activate ffs
    cd F:/Research/02_Projects/SRTP/Reproduction

    # Test first 10 frames
    python scripts/run_ffs_batch.py --clip clip03 --end_frame 10

    # Full sequence (will skip existing by default)
    python scripts/run_ffs_batch.py --clip clip03

    # Force overwrite existing depth
    python scripts/run_ffs_batch.py --clip clip03 --overwrite

    # Dry run (show what would be done)
    python scripts/run_ffs_batch.py --clip clip03 --dry_run
"""

import argparse, os, sys, time, glob, logging

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(code_dir)

# Ensure FFS is on path
sys.path.insert(0, os.path.join(repo_dir, 'Fast-FoundationStereo'))

import numpy as np
import cv2
import torch
from run_demo_save_depth import load_ffs_model, run_ffs_single, read_K_from_3line, read_baseline_from_calib
from Utils import set_logging_format, set_seed


def depth_npy_to_png(depth_m, output_path):
    """Convert depth_meter float32 (m) numpy array to uint16 mm PNG."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    depth_m[~np.isfinite(depth_m)] = 0
    depth_m[depth_m < 0] = 0
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, depth_mm)
    valid = depth_m[depth_m > 0.001]
    return {
        'path': output_path,
        'shape': depth_mm.shape,
        'depth_min_m': float(valid.min()) if len(valid) > 0 else None,
        'depth_max_m': float(depth_m.max()),
    }


def main():
    parser = argparse.ArgumentParser(description='Batch FFS depth generation')
    parser.add_argument('--clip', type=str, default='clip03')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=-1,
                        help='End frame (exclusive); -1 = all')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing depth PNGs')
    parser.add_argument('--dry_run', action='store_true',
                        help='List what would be done without running')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='FFS model .pth path')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Image downscale factor (1.0 = full 1920x1080)')
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--zfar', type=float, default=100)
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save disp.npy, cloud.ply, etc. in ffs/raw/')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    data_dir = os.path.join(repo_dir, 'data', args.clip)
    rgb_dir = os.path.join(data_dir, 'rgb')
    right_dir = os.path.join(data_dir, 'right')
    K_file = os.path.join(data_dir, 'ffs', 'cam_K.txt')
    calib_file = os.path.join(data_dir, 'calib.json')
    depth_out_dir = os.path.join(data_dir, 'ffs', 'depth')

    # Gather sorted frame list
    left_frames = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
    if not left_frames:
        left_frames = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    if not left_frames:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")

    total = len(left_frames)
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame > 0 else total
    end_frame = min(end_frame, total)

    # Load calibration
    K = read_K_from_3line(K_file)
    baseline = read_baseline_from_calib(calib_file)
    logging.info(f"K:\n{K}")
    logging.info(f"Baseline: {baseline:.4f}m")

    # Load model once
    if args.model_dir is None:
        args.model_dir = os.path.join(repo_dir, 'Fast-FoundationStereo', 'weights',
                                      '23-36-37', 'model_best_bp2_serialize.pth')
    if args.dry_run:
        logging.info(f"[DRY RUN] Would load model from {args.model_dir}")
    else:
        model, cfg = load_ffs_model(args.model_dir)

    # Count work to do
    todo = 0
    skipped = 0
    for i in range(start_frame, end_frame):
        stem = os.path.splitext(os.path.basename(left_frames[i]))[0]
        right_file = os.path.join(right_dir, f'{stem}.jpg')
        if not os.path.exists(right_file):
            right_file = os.path.join(right_dir, f'{stem}.png')
        if not os.path.exists(right_file):
            logging.warning(f"Frame {stem}: no right image, skip")
            continue
        depth_png = os.path.join(depth_out_dir, f'{stem}.png')
        if os.path.exists(depth_png) and not args.overwrite:
            skipped += 1
        else:
            todo += 1

    logging.info(f"Frames: {start_frame}–{end_frame - 1} / {total} total")
    logging.info(f"To process: {todo}, skipped (existing): {skipped}")

    if args.dry_run:
        for i in range(start_frame, end_frame):
            stem = os.path.splitext(os.path.basename(left_frames[i]))[0]
            depth_png = os.path.join(depth_out_dir, f'{stem}.png')
            if os.path.exists(depth_png) and not args.overwrite:
                logging.info(f"  [{i}] {stem} — SKIP (exists)")
            else:
                logging.info(f"  [{i}] {stem} — WOULD PROCESS")
        logging.info("Dry run done.")
        return

    if todo == 0:
        logging.info("All frames already processed. Use --overwrite to re-run.")
        return

    # Process
    t_start = time.time()
    done = 0
    for i in range(start_frame, end_frame):
        stem = os.path.splitext(os.path.basename(left_frames[i]))[0]
        left_file = left_frames[i]
        right_file = os.path.join(right_dir, f'{stem}.jpg')
        if not os.path.exists(right_file):
            right_file = os.path.join(right_dir, f'{stem}.png')
        if not os.path.exists(right_file):
            continue

        depth_png = os.path.join(depth_out_dir, f'{stem}.png')
        if os.path.exists(depth_png) and not args.overwrite:
            continue

        logging.info(f"[{done + 1}/{todo}] Frame {i} ({stem})")

        if args.save_intermediate:
            raw_dir = os.path.join(data_dir, 'ffs', 'raw', stem)
        else:
            raw_dir = os.path.join(data_dir, 'ffs', 'raw', '_tmp')
        os.makedirs(raw_dir, exist_ok=True)

        result = run_ffs_single(
            model=model, cfg=cfg,
            left_file=left_file, right_file=right_file,
            K=K, baseline=baseline,
            out_dir=raw_dir,
            scale=args.scale,
            valid_iters=args.valid_iters,
            max_disp=args.max_disp,
            zfar=args.zfar,
            remove_invisible=True,
            get_pc=args.save_intermediate,
            denoise_cloud=False,
        )

        # Convert depth to uint16 mm PNG
        info = depth_npy_to_png(result['depth_meter'], depth_png)
        logging.info(f"  -> {depth_png}  "
                     f"(depth: {info['depth_min_m']:.3f}–{info['depth_max_m']:.3f}m)")

        # Clean up temp dir
        if not args.save_intermediate:
            import shutil
            if os.path.exists(raw_dir):
                shutil.rmtree(raw_dir, ignore_errors=True)

        done += 1

    t_elapsed = time.time() - t_start
    logging.info(f"Done. Processed {done} frames in {t_elapsed:.1f}s "
                 f"({t_elapsed / done:.1f}s/frame)")
    logging.info(f"Depth PNGs: {depth_out_dir}/")


if __name__ == '__main__':
    main()
