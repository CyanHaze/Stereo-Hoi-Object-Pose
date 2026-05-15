#!/usr/bin/env python
"""
Multi-view pose fusion: combines left and right FoundationPose tracking results
into a single pose sequence in the left-camera coordinate frame.

Input:
    foundationpose_v2/run/ob_in_cam/*.txt         (left-camera poses)
    foundationpose_v2/run_right/ob_in_cam/*.txt   (right-camera poses)
    calib.json                                    (baseline_m)
    ffs/cam_K.txt                                 (intrinsics)

Output:
    foundationpose_v2/fused/ob_in_cam/*.txt       (fused poses, left-cam frame)
    foundationpose_v2/fused/track_vis/*.png       (mesh render + box + axis)
    foundationpose_v2/fused/video_frames/*.png    (RGB overlay + box + axis)

Usage:
    # Fusion only (no GPU needed)
    python scripts/run_fusion.py --clip clip03

    # Fusion + visualization (needs FoundationPose Docker container)
    python scripts/run_fusion.py --clip clip03 --vis

    # Ablation: left-only baseline
    python scripts/run_fusion.py --clip clip03 --method left_only --vis

    # With temporal smoothing
    python scripts/run_fusion.py --clip clip03 --smooth 5 --vis
"""

import argparse, os, sys, json, glob, logging

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(code_dir)

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def build_left_from_right(baseline_m):
    """Right camera -> left camera transform for rectified stereo.

    In rectified stereo the right camera is at [baseline, 0, 0] in the
    left-camera frame.  Hence T_left_from_right = translate(+baseline, 0, 0).
    """
    T = np.eye(4)
    T[0, 3] = baseline_m
    return T


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_poses(txt_dir, id_strs):
    """Load 4x4 pose matrices from a directory of .txt files.

    Returns (poses, missing_ids).
    """
    N = len(id_strs)
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    missing = np.zeros(N, dtype=bool)
    for i, sid in enumerate(id_strs):
        path = os.path.join(txt_dir, f'{sid}.txt')
        if os.path.exists(path):
            poses[i] = np.loadtxt(path).reshape(4, 4)
        else:
            missing[i] = True
    return poses, missing


def save_poses(txt_dir, id_strs, poses):
    """Save 4x4 poses as .txt files."""
    os.makedirs(txt_dir, exist_ok=True)
    for i, sid in enumerate(id_strs):
        np.savetxt(os.path.join(txt_dir, f'{sid}.txt'), poses[i].reshape(4, 4))


# ---------------------------------------------------------------------------
# Pose fusion
# ---------------------------------------------------------------------------

def fuse_average(poses_left, poses_right, valid):
    """Equal-weight fusion.

    Rotation: mean quaternion (normalised).
    Translation: arithmetic mean.
    """
    N = len(valid)
    fused = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        if not valid[i]:
            if not np.all(poses_left[i] == 0):
                fused[i] = poses_left[i]
            else:
                fused[i] = poses_right[i]
            continue

        r_l = R.from_matrix(poses_left[i, :3, :3])
        r_r = R.from_matrix(poses_right[i, :3, :3])

        q = (r_l.as_quat() + r_r.as_quat()) / 2.0
        q /= np.linalg.norm(q)
        r_fused = R.from_quat(q)

        t_fused = (poses_left[i, :3, 3] + poses_right[i, :3, 3]) / 2.0

        fused[i, :3, :3] = r_fused.as_matrix()
        fused[i, :3, 3] = t_fused
        fused[i, 3, 3] = 1.0
    return fused


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_poses(poses, window=5):
    """Moving-average smooth a pose sequence in quaternion + translation space."""
    if window <= 1:
        return poses.copy()

    from scipy.ndimage import uniform_filter1d

    N = poses.shape[0]
    quats = np.zeros((N, 4))
    trans = np.zeros((N, 3))
    for i in range(N):
        quats[i] = R.from_matrix(poses[i, :3, :3]).as_quat()
        trans[i] = poses[i, :3, 3]

    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    smoothed_quat = np.zeros_like(quats)
    smoothed_trans = np.zeros_like(trans)
    for d in range(4):
        smoothed_quat[:, d] = uniform_filter1d(quats[:, d], size=window)
    for d in range(3):
        smoothed_trans[:, d] = uniform_filter1d(trans[:, d], size=window)

    norms = np.linalg.norm(smoothed_quat, axis=1, keepdims=True)
    smoothed_quat /= norms

    smoothed = poses.copy()
    for i in range(N):
        smoothed[i, :3, :3] = R.from_quat(smoothed_quat[i]).as_matrix()
        smoothed[i, :3, 3] = smoothed_trans[i]
    return smoothed


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_stats(poses_left, poses_right, fused, valid):
    """Print per-frame comparison statistics."""
    diffs_LR, diffs_LF = [], []
    for i in range(len(valid)):
        if not valid[i]:
            continue
        t_l = poses_left[i, :3, 3]
        t_r = poses_right[i, :3, 3]
        t_f = fused[i, :3, 3]
        diffs_LR.append(np.linalg.norm(t_l - t_r))
        diffs_LF.append(np.linalg.norm(t_l - t_f))

    diffs_LR = np.array(diffs_LR)
    diffs_LF = np.array(diffs_LF)
    logging.info(f"Frames: {len(valid)} total, {valid.sum()} fused, "
                 f"{(~valid).sum()} missing on one side")
    logging.info(f"Left <-> Right translation discrepancy (mm): "
                 f"mean={diffs_LR.mean()*1000:.1f}  max={diffs_LR.max()*1000:.1f}")
    logging.info(f"Left <-> Fused translation discrepancy (mm): "
                 f"mean={diffs_LF.mean()*1000:.1f}  max={diffs_LF.max()*1000:.1f}")


# ---------------------------------------------------------------------------
# Visualization (requires FoundationPose + CUDA)
# ---------------------------------------------------------------------------

def _try_import_fp():
    """Try importing FoundationPose rendering modules.  Returns (ok, modules_dict)."""
    try:
        sys.path.insert(0, os.path.join(repo_dir, 'FoundationPose'))
        import estimater
        import trimesh
        import torch
        import nvdiffrast.torch as dr
        import imageio
        return True, {
            'estimater': estimater, 'trimesh': trimesh,
            'torch': torch, 'dr': dr, 'imageio': imageio,
        }
    except ImportError as e:
        logging.warning(f"Cannot import FoundationPose / CUDA modules: {e}")
        logging.warning("Visualization disabled. Run inside the FP Docker container for --vis.")
        return False, {}


def generate_vis(data_dir, id_strs, fused_poses, args):
    """Generate track_vis and video_frames for fused poses.

    Mirrors the debug=2 rendering from run_fp_track.py.
    """
    ok, m = _try_import_fp()
    if not ok:
        return

    estimater = m['estimater']
    trimesh = m['trimesh']
    torch = m['torch']
    dr_mod = m['dr']
    imageio = m['imageio']

    # ---- mesh ----
    mesh_file = os.path.join(data_dir, 'mesh', 'clean_mesh.obj')
    if not os.path.exists(mesh_file):
        logging.warning(f"Mesh not found: {mesh_file}")
        return

    scale_file = os.path.join(data_dir, 'foundationpose', 'run', 'scales', 'unified_scale.txt')
    mesh_scale = float(open(scale_file).read().strip()) if os.path.exists(scale_file) else 1.0
    logging.info(f"[vis] Mesh scale: {mesh_scale}")

    mesh = trimesh.load(mesh_file)
    mesh.vertices = mesh.vertices * mesh_scale
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ---- intrinsics & image resolution ----
    K_path = os.path.join(data_dir, 'ffs', 'cam_K.txt')
    K_full = np.loadtxt(K_path).reshape(3, 3)

    rgb_dir = os.path.join(data_dir, 'rgb')
    sample_img = cv2.imread(os.path.join(rgb_dir, f'{id_strs[0]}.jpg'))
    if sample_img is None:
        sample_img = cv2.imread(os.path.join(rgb_dir, f'{id_strs[0]}.png'))
    H_orig, W_orig = sample_img.shape[:2]

    shorter_side = args.shorter_side
    downscale = shorter_side / min(H_orig, W_orig)
    H, W = int(H_orig * downscale), int(W_orig * downscale)
    K = K_full.copy()
    K[:2] *= downscale

    # ---- FoundationPose (minimal, for mesh_tensors + glctx) ----
    scorer = estimater.ScorePredictor()
    refiner = estimater.PoseRefinePredictor()
    glctx = dr_mod.RasterizeCudaContext()
    est = estimater.FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir=os.path.join(data_dir, 'foundationpose_v2', 'fused'),
        debug=0, glctx=glctx,
    )
    logging.info("[vis] FoundationPose renderer ready")

    # ---- render each frame ----
    out_vis = os.path.join(data_dir, 'foundationpose_v2', 'fused', 'track_vis')
    out_vf = os.path.join(data_dir, 'foundationpose_v2', 'fused', 'video_frames')
    os.makedirs(out_vis, exist_ok=True)
    os.makedirs(out_vf, exist_ok=True)

    for i, sid in enumerate(id_strs):
        pose = fused_poses[i]

        # Read + resize left RGB
        rgb_path = os.path.join(rgb_dir, f'{sid}.jpg')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(rgb_dir, f'{sid}.png')
        color = cv2.imread(rgb_path)
        if color is None:
            logging.warning(f"[vis] Frame {sid}: cannot read RGB, skip")
            continue
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)

        center_pose = pose @ np.linalg.inv(to_origin)

        # --- track_vis: black-background mesh render ---
        H8 = (H + 7) // 8 * 8
        W8 = (W + 7) // 8 * 8
        K8 = K.copy()
        K8[0] *= W8 / W
        K8[1] *= H8 / H

        ob_in_cam_t = torch.as_tensor(pose, device='cuda', dtype=torch.float).reshape(1, 4, 4)
        rendered, _, _ = estimater.nvdiffrast_render(
            K=K8, H=H8, W=W8,
            ob_in_cams=ob_in_cam_t, glctx=est.glctx,
            mesh_tensors=est.mesh_tensors, mesh=mesh,
        )
        rendered_np = (rendered[0].data.cpu().numpy() * 255).astype(np.uint8)
        rendered_np = rendered_np[:H, :W, :3].copy()

        vis_track = estimater.draw_posed_3d_box(K, img=rendered_np, ob_in_cam=center_pose, bbox=bbox)
        vis_track = estimater.draw_xyz_axis(vis_track, ob_in_cam=center_pose, scale=0.1,
                                            K=K, thickness=3, transparency=0, is_input_rgb=True)
        imageio.imwrite(os.path.join(out_vis, f'{sid}.png'), vis_track)

        # --- video_frames: RGB overlay ---
        vis_rgb = estimater.draw_posed_3d_box(K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
        vis_rgb = estimater.draw_xyz_axis(vis_rgb, ob_in_cam=center_pose, scale=0.1,
                                          K=K, thickness=3, transparency=0, is_input_rgb=True)
        imageio.imwrite(os.path.join(out_vf, f'{sid}.png'), vis_rgb[..., ::-1])  # BGR->RGB

        if (i + 1) % 50 == 0:
            logging.info(f"[vis] {i + 1}/{len(id_strs)} frames rendered")

    logging.info(f"[vis] Done -> {out_vis}/  +  {out_vf}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multi-view pose fusion for stereo FoundationPose')
    parser.add_argument('--clip', type=str, default='clip03')
    parser.add_argument('--method', type=str, default='average',
                        choices=['average', 'left_only', 'right_only'],
                        help='Fusion strategy (default: average)')
    parser.add_argument('--smooth', type=int, default=0,
                        help='Temporal smoothing window (odd; 0=off; e.g. 5)')
    parser.add_argument('--vis', action='store_true',
                        help='Generate track_vis + video_frames (requires FoundationPose/CUDA)')
    parser.add_argument('--shorter_side', type=int, default=800,
                        help='Resize shorter side for vis rendering (default: 800)')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    data_dir = os.path.join(repo_dir, 'data', args.clip)
    left_dir = os.path.join(data_dir, 'foundationpose_v2', 'run', 'ob_in_cam')
    right_dir = os.path.join(data_dir, 'foundationpose_v2', 'run_right', 'ob_in_cam')
    out_dir = os.path.join(data_dir, 'foundationpose_v2', 'fused', 'ob_in_cam')

    # ----- calibration -----
    calib_path = os.path.join(data_dir, 'calib.json')
    with open(calib_path) as f:
        calib = json.load(f)
    baseline_m = float(calib['baseline_m'])
    logging.info(f"Baseline: {baseline_m * 1000:.2f} mm")

    T_left_from_right = build_left_from_right(baseline_m)

    # ----- frame list -----
    left_files = sorted(glob.glob(os.path.join(left_dir, '*.txt')))
    if not left_files:
        raise FileNotFoundError(f"No pose files in {left_dir}")
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in left_files]
    logging.info(f"Found {len(id_strs)} pose files in left directory")

    end = args.end_frame if args.end_frame > 0 else len(id_strs)
    id_strs = id_strs[args.start_frame:end]
    logging.info(f"Processing frames [{args.start_frame}, {end}) - {len(id_strs)} frames")

    # ----- load -----
    poses_left, miss_left = load_poses(left_dir, id_strs)
    poses_right_raw, miss_right = load_poses(right_dir, id_strs)

    poses_right = poses_right_raw.copy()
    for i in range(len(id_strs)):
        if not miss_right[i]:
            poses_right[i] = T_left_from_right @ poses_right_raw[i]

    valid = ~miss_left & ~miss_right

    # ----- fuse -----
    if args.method == 'average':
        fused = fuse_average(poses_left, poses_right, valid)
    elif args.method == 'left_only':
        fused = poses_left.copy()
    elif args.method == 'right_only':
        fused = poses_right.copy()
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print_stats(poses_left, poses_right, fused, valid)

    # ----- temporal smoothing -----
    if args.smooth > 0:
        w = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
        logging.info(f"Temporal smoothing: window={w}")
        fused = smooth_poses(fused, window=w)

    # ----- save poses -----
    save_poses(out_dir, id_strs, fused)
    logging.info(f"Saved {len(id_strs)} fused poses -> {out_dir}")

    # ----- visualization -----
    if args.vis:
        generate_vis(data_dir, id_strs, fused, args)


if __name__ == '__main__':
    main()
