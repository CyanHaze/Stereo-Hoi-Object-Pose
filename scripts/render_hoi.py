#!/usr/bin/env python
"""
Combined hand + object rendering for hand-object interaction visualization.

Renders both the hand MANO mesh and the object mesh (with oriented bbox)
overlaid on the RGB frames, using FFS depth-aligned metric 3D coordinates.

Requires: WiloR inference results (run_wilor_hand.py) + FoundationPose tracking
          results (run_fp_track.py) to already exist on disk.

Output:
    data/<clip>/hoi/video_frames/  — combined overlay PNGs per frame

Usage:
    conda activate diffusion
    cd F:/Research/02_Projects/SRTP/Reproduction
    python scripts/render_hoi.py --clip clip03 --end_frame 10
    python scripts/render_hoi.py --clip clip03  # full sequence
    python scripts/render_hoi.py --clip clip03 --fps 30  # also compose MP4
"""

import argparse, os, sys, glob, logging, time
import numpy as np
import cv2

REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, 'WiLoR'))


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def load_obj_verts_faces(obj_path):
    """Read vertices and faces from .obj, converting quads → triangles."""
    verts = []
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('f '):
                parts = [p.split('/')[0] for p in line.strip().split()[1:]]
                idxs = [int(p) - 1 for p in parts]
                if len(idxs) == 3:
                    faces.append(idxs)
                elif len(idxs) == 4:
                    faces.append([idxs[0], idxs[1], idxs[2]])
                    faces.append([idxs[0], idxs[2], idxs[3]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def compute_oriented_bbox(verts):
    """Compute oriented bounding box.  Returns (to_origin, bbox(2,3))."""
    try:
        import trimesh
        m = trimesh.Trimesh(verts, faces=np.zeros((0, 3)))
        to_origin, extents = trimesh.bounds.oriented_bounds(m)
    except ImportError:
        center = (verts.min(axis=0) + verts.max(axis=0)) / 2
        extents = verts.max(axis=0) - verts.min(axis=0)
        to_origin = np.eye(4)
        to_origin[:3, 3] = center
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    return to_origin, bbox


# ---------------------------------------------------------------------------
# 3D bbox drawing (from FoundationPose Utils, pure numpy)
# ---------------------------------------------------------------------------

def _to_homo(pts):
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def _line(start, end):
        pts = np.stack((start, end)).reshape(-1, 3)
        pts = (ob_in_cam @ _to_homo(pts).T).T[:, :3]
        proj = (K @ pts.T).T
        uv = np.round(proj[:, :2] / proj[:, 2].reshape(-1, 1)).astype(int)
        cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color,
                 thickness=linewidth, lineType=cv2.LINE_AA)

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            _line(np.array([xmin, y, z]), np.array([xmax, y, z]))
    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            _line(np.array([x, ymin, z]), np.array([x, ymax, z]))
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            _line(np.array([x, y, zmin]), np.array([x, y, zmax]))


# ---------------------------------------------------------------------------
# Mesh rendering
# ---------------------------------------------------------------------------

def render_mesh(img_bgr, verts_cam, faces, K, color, alpha=0.5):
    """Filled-triangle overlay with alpha blending."""
    H, W = img_bgr.shape[:2]
    p = K @ verts_cam.T
    z = p[2]
    p = (p[:2] / (z + 1e-9)).T

    z_ok = (z[faces] > 0.01).all(axis=1)
    in_img = ((p[faces] >= -50) & (p[faces] < np.array([W + 50, H + 50]))).all(axis=(1, 2))
    valid = z_ok & in_img
    if valid.sum() == 0:
        return img_bgr

    tri_pts = p[faces[valid]].astype(np.int32)
    overlay = img_bgr.copy()
    cv2.fillPoly(overlay, tri_pts, tuple(int(c) for c in color), lineType=cv2.LINE_AA)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, tri_pts, 255, lineType=cv2.LINE_AA)
    mb = mask > 0
    img_bgr[mb] = (img_bgr[mb] * (1 - alpha) + overlay[mb] * alpha).astype(np.uint8)
    return img_bgr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='HOI combined rendering')
    parser.add_argument('--clip', type=str, default='clip03')
    parser.add_argument('--camera', type=str, default='left')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=-1)
    parser.add_argument('--fps', type=int, default=0,
                        help='If >0, compose MP4 with ffmpeg after rendering')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    data_dir = os.path.join(REPO_DIR, 'data', args.clip)
    hand_dir = os.path.join(data_dir, 'wilor', args.camera)
    obj_pose_dir = os.path.join(data_dir, 'foundationpose_v2',
                                'fused' if args.camera == 'left' else 'run_right',
                                'ob_in_cam')
    rgb_dir = os.path.join(data_dir, 'rgb' if args.camera == 'left' else 'right')
    out_dir = os.path.join(data_dir, 'hoi', 'video_frames')
    os.makedirs(out_dir, exist_ok=True)

    # --- Camera ---
    K = np.loadtxt(os.path.join(data_dir, 'ffs', 'cam_K.txt')).reshape(3, 3)
    # Virtual K for hand rendering (WiLoR camera model)
    sample_imgs = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
    if not sample_imgs:
        sample_imgs = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    img = cv2.imread(sample_imgs[0])
    img_h, img_w = img.shape[:2]
    f_virt = 5000.0 / 256.0 * max(img_w, img_h)
    K_virt = np.array([[f_virt, 0, img_w / 2.],
                       [0, f_virt, img_h / 2.],
                       [0, 0, 1]], dtype=np.float32)

    # --- Object mesh ---
    mesh_file = os.path.join(data_dir, 'mesh', 'clean_mesh.obj')
    obj_verts, obj_faces = load_obj_verts_faces(mesh_file)
    scale_file = os.path.join(data_dir, 'foundationpose', 'run', 'scales',
                              'unified_scale.txt')
    mesh_scale = float(open(scale_file).read().strip()) if os.path.exists(scale_file) else 1.0
    obj_verts *= mesh_scale
    to_origin, bbox = compute_oriented_bbox(obj_verts)
    logging.info(f"Object mesh: {len(obj_verts)}v, {len(obj_faces)}f, "
                 f"scale={mesh_scale:.4f}, bbox diag={np.linalg.norm(bbox[1]-bbox[0]):.3f}m")

    # --- MANO faces (constant; load from cached npy or WiLoR model) ---
    mano_faces_path = os.path.join(REPO_DIR, 'data', 'mano_faces.npy')
    if os.path.exists(mano_faces_path):
        mano_faces = np.load(mano_faces_path)
    else:
        wilor_dir = os.path.join(REPO_DIR, 'WiLoR')
        _cwd = os.getcwd()
        os.chdir(wilor_dir)
        sys.path.insert(0, wilor_dir)
        import torch
        _orig_ld = torch.load
        def _pld(*a, **kw): kw.setdefault('weights_only', False); return _orig_ld(*a, **kw)
        torch.load = _pld
        from wilor.models import load_wilor
        m, _ = load_wilor('./pretrained_models/wilor_final.ckpt',
                          './pretrained_models/model_config.yaml')
        mano_faces = m.mano.faces
        np.save(mano_faces_path, mano_faces)
        os.chdir(_cwd)
        logging.info("Cached MANO faces to data/mano_faces.npy")

    # --- Frame list ---
    hand_files = sorted(glob.glob(os.path.join(hand_dir, '*.npz')))
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in hand_files]
    total = len(id_strs)
    start, end = args.start_frame, (args.end_frame if args.end_frame > 0 else total)
    end = min(end, total)
    id_strs = id_strs[start:end]

    todo = sum(1 for sid in id_strs
               if args.overwrite or
               not os.path.exists(os.path.join(out_dir, f'{sid}.png')))

    if todo == 0:
        logging.info("All frames already rendered.")
        if args.fps > 0:
            compose_video_ffmpeg(out_dir, os.path.join(data_dir, 'hoi', 'track.mp4'), args.fps)
        return

    logging.info(f"Rendering {todo} frames → {out_dir}/")
    t0 = time.time()

    for i, sid in enumerate(id_strs):
        out_path = os.path.join(out_dir, f'{sid}.png')
        if os.path.exists(out_path) and not args.overwrite:
            continue

        # RGB
        img_path = os.path.join(rgb_dir, f'{sid}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(rgb_dir, f'{sid}.png')
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # --- Render object ---
        obj_pose_path = os.path.join(obj_pose_dir, f'{sid}.txt')
        if os.path.exists(obj_pose_path):
            pose = np.loadtxt(obj_pose_path).reshape(4, 4)
            obj_verts_cam = (pose[:3, :3] @ obj_verts.T).T + pose[:3, 3]
            # Object color: green
            img_bgr = render_mesh(img_bgr, obj_verts_cam, obj_faces, K,
                                   color=(100, 200, 80), alpha=0.45)
            # Object bbox
            center_pose = pose @ np.linalg.inv(to_origin)
            draw_posed_3d_box(K, img_bgr, center_pose, bbox,
                               line_color=(0, 220, 80), linewidth=2)

        # --- Render hands ---
        hand_path = os.path.join(hand_dir, f'{sid}.npz')
        if os.path.exists(hand_path):
            hd = np.load(hand_path)
            N = len(hd['verts_virt'])
            for n in range(N):
                is_r = hd['is_right'][n]
                color = (255, 128, 0) if is_r else (0, 220, 220)  # orange / yellow-cyan
                w = hd['wrist_3d'][n]
                ok = hd['depth_ok'][n]
                # Use virtual-camera rendering for correct 2D projection
                img_bgr = render_mesh(img_bgr, hd['verts_virt'][n],
                                       mano_faces, K_virt, color=color, alpha=0.45)
                # Draw wrist dot
                if ok and np.linalg.norm(w) > 0.001:
                    pw = K @ w
                    pw = (pw[:2] / pw[2]).astype(int)
                    cv2.circle(img_bgr, tuple(pw.tolist()), 5,
                               (0, 0, 255), -1, lineType=cv2.LINE_AA)

        cv2.imwrite(out_path, img_bgr)

        if (i + 1) % 50 == 0:
            e = time.time() - t0
            logging.info(f"  {i + 1}/{todo}  ({e:.0f}s, {(i + 1) / e:.1f} fps)")

    e = time.time() - t0
    logging.info(f"Done: {todo} frames in {e:.0f}s → {out_dir}/")

    # MP4
    if args.fps > 0:
        compose_video_ffmpeg(out_dir, os.path.join(data_dir, 'hoi', 'track.mp4'),
                             args.fps)


def compose_video_ffmpeg(frame_dir, output_mp4, fps):
    import subprocess
    files = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
    if not files:
        return
    list_path = output_mp4.replace('.mp4', '_concat.txt')
    with open(list_path, 'w') as fl:
        for p in files:
            fl.write(f"file '{p.replace(chr(92), '/')}'\n")
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-r', str(fps),
           '-i', list_path, '-c:v', 'libx264', '-crf', '23',
           '-pix_fmt', 'yuv420p', output_mp4]
    logging.info(f"Composing {len(files)} frames → {output_mp4}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        os.remove(list_path)
        logging.info(f"  → {output_mp4} ({os.path.getsize(output_mp4) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
