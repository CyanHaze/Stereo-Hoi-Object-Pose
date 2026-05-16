#!/usr/bin/env python
"""
Build comparison videos from FoundationPose tracking results.

Two modes:
  render  — generate video_frames from ob_in_cam/*.txt poses (pure numpy+cv2, no FP import)
  compose — ffmpeg video_frames/*.png → MP4
  both    — render + compose (default)

Output videos (placed alongside video_frames/):
  foundationpose_v2/run/track.mp4          (left-only)
  foundationpose_v2/run_right/track.mp4    (right-only)
  foundationpose_v2/fused/track.mp4        (fused)
  foundationpose_v2/comparison_left_fused.mp4  (left-only | fused, side-by-side)

Usage:
  # Full pipeline (render missing frames + compose videos)
  python scripts/build_comparison_video.py --clip clip03

  # Compose only (video_frames already complete, no mesh needed)
  python scripts/build_comparison_video.py --clip clip03 --mode compose

  # Only specific views
  python scripts/build_comparison_video.py --clip clip03 --views left,fused

  # Custom FPS / frame range
  python scripts/build_comparison_video.py --clip clip03 --fps 30 --start_frame 0 --end_frame 200
"""

import argparse, os, sys, json, glob, subprocess, logging
import numpy as np
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(code_dir)


# ---------------------------------------------------------------------------
# Inlined from FoundationPose/Utils.py (pure numpy+cv2, no torch dependency)
# ---------------------------------------------------------------------------

def _to_homo(pts):
    assert len(pts.shape) == 2, f'pts.shape: {pts.shape}'
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)


def _project_3d_to_2d(pt, K, ob_in_cam):
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(color, ob_in_cam, scale=0.1, K=None, thickness=3,
                  transparency=0, is_input_rgb=False):
    if K is None:
        K = np.eye(3)
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float)
    yy = np.array([0, 1, 0, 1]).astype(float)
    zz = np.array([0, 0, 1, 1]).astype(float)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = tuple(_project_3d_to_2d(np.array([0, 0, 0, 1]), K, ob_in_cam))
    xx = tuple(_project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(_project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(_project_3d_to_2d(zz, K, ob_in_cam))
    arrow_len = 0
    tmp = color.copy()
    for pt, clr in [(xx, (0, 0, 255)), (yy, (0, 255, 0)), (zz, (255, 0, 0))]:
        tmp1 = tmp.copy()
        tmp1 = cv2.arrowedLine(tmp1, origin, pt, color=clr, thickness=thickness,
                               line_type=cv2.LINE_AA, tipLength=arrow_len)
        mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
        tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    return tmp


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def _draw_line3d(start, end, img):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (ob_in_cam @ _to_homo(pts).T).T[:, :3]
        projected = (K @ pts.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(int)
        return cv2.line(img, uv[0].tolist(), uv[1].tolist(),
                        color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            img = _draw_line3d(start, start + np.array([xmax - xmin, 0, 0]), img)
    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            img = _draw_line3d(start, start + np.array([0, ymax - ymin, 0]), img)
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            img = _draw_line3d(start, start + np.array([0, 0, zmax - zmin]), img)
    return img


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_poses(txt_dir, id_strs):
    """Load 4x4 poses. Returns (N,4,4); rows with missing files stay zeros."""
    poses = np.zeros((len(id_strs), 4, 4), dtype=np.float64)
    for i, sid in enumerate(id_strs):
        path = os.path.join(txt_dir, f'{sid}.txt')
        if os.path.exists(path):
            poses[i] = np.loadtxt(path).reshape(4, 4)
    return poses


def compute_bbox(mesh_file, mesh_scale=1.0):
    """Return (to_origin, bbox_2x3) for a mesh, with pure numpy fallback."""
    try:
        import trimesh
        mesh = trimesh.load(mesh_file)
        mesh.vertices = mesh.vertices * mesh_scale
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    except ImportError:
        logging.warning("trimesh not available, using AABB from obj vertices")
        vertices = _read_obj_vertices(mesh_file) * mesh_scale
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        to_origin = np.eye(4)
        to_origin[:3, 3] = center
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    return to_origin, bbox


def _read_obj_vertices(obj_path):
    """Read vertex positions from a .obj file (pure python, no trimesh)."""
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.strip().split()[1:4]])
    return np.array(verts, dtype=np.float64)


def get_frame_list(data_dir, start_frame, end_frame):
    """Return sorted id_strs and rgb_dir from data_dir/rgb/."""
    rgb_dir = os.path.join(data_dir, 'rgb')
    color_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
    if not color_files:
        color_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in color_files]
    end = end_frame if end_frame > 0 else len(id_strs)
    id_strs = id_strs[start_frame:min(end, len(id_strs))]
    return id_strs, rgb_dir


# ---------------------------------------------------------------------------
# Render: ob_in_cam/*.txt + rgb/*.jpg → video_frames/*.png
# ---------------------------------------------------------------------------

def render_frames(data_dir, pose_dir, out_dir, id_strs, K, to_origin, bbox,
                  left_color=(0, 255, 0), overwrite=False):
    """Render RGB+box+axis frames. Skips existing unless overwrite=True."""
    rgb_dir = os.path.join(data_dir, 'rgb')
    os.makedirs(out_dir, exist_ok=True)

    done, skipped, errored = 0, 0, 0
    for i, sid in enumerate(id_strs):
        out_path = os.path.join(out_dir, f'{sid}.png')
        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            continue

        pose = load_poses(pose_dir, [sid])[0]
        if np.all(pose == 0) or not np.isfinite(pose).all():
            errored += 1
            continue

        rgb_path = os.path.join(rgb_dir, f'{sid}.jpg')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(rgb_dir, f'{sid}.png')
        color = cv2.imread(rgb_path)
        if color is None:
            errored += 1
            continue

        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, color.copy(), center_pose, bbox,
                                line_color=left_color)
        vis = draw_xyz_axis(vis, center_pose, scale=0.1, K=K,
                            thickness=3, transparency=0, is_input_rgb=False)
        cv2.imwrite(out_path, vis)
        done += 1

        if done % 100 == 0:
            logging.info(f"  render {done}/{len(id_strs)}")

    logging.info(f"  -> {out_dir}: {done} rendered, {skipped} skipped, {errored} errors")


# ---------------------------------------------------------------------------
# Compose: video_frames/*.png → .mp4 (ffmpeg)
# ---------------------------------------------------------------------------

def compose_video(frame_dir, output_mp4, id_strs, fps=30, crf=23):
    """Run ffmpeg to compose PNG sequence → MP4.

    Only frames whose stem is in id_strs are included (respects start/end range).
    Uses a temporary concat-file list so stray frames outside the range are ignored.
    """
    if not os.path.isdir(frame_dir):
        logging.warning(f"  frame dir not found: {frame_dir}")
        return False

    # Collect only frames in the requested range, in order
    file_list = []
    missing = 0
    for sid in id_strs:
        p = os.path.join(frame_dir, f'{sid}.png')
        if os.path.exists(p):
            file_list.append(p)
        else:
            missing += 1

    if len(file_list) == 0:
        logging.warning(f"  no frames for {len(id_strs)} requested ids in {frame_dir}")
        return False
    if missing > 0:
        logging.info(f"  {missing}/{len(id_strs)} frames missing, will use {len(file_list)} available")

    # Write concat file list (use forward slashes for ffmpeg on Windows)
    list_path = output_mp4.replace('.mp4', '_concat.txt')
    with open(list_path, 'w') as fl:
        for p in file_list:
            fl.write(f"file '{p.replace(chr(92), '/')}'\n")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-r', str(fps),
        '-i', list_path,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-vsync', 'vfr',
        output_mp4,
    ]
    logging.info(f"  ffmpeg: {len(file_list)} frames -> {output_mp4}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            os.remove(list_path)
            logging.info(f"  -> {output_mp4}  ({os.path.getsize(output_mp4)/1e6:.1f} MB)")
            return True
        else:
            logging.error(f"  ffmpeg failed: {r.stderr[-400:]}")
            return False
    except FileNotFoundError:
        logging.error("  ffmpeg not found — install ffmpeg and retry")
        return False


def compose_comparison(left_mp4, right_mp4, output_mp4, label_left="Left Only", label_right="Fused"):
    """hstack two videos into a side-by-side comparison."""
    if not os.path.exists(left_mp4) or not os.path.exists(right_mp4):
        logging.warning("  one or both input videos missing, skip comparison")
        return False

    filt = (
        f"[0:v]drawtext=text='{label_left}':fontcolor=white:fontsize=28:"
        f"x=10:y=h-th-10,format=yuv420p[left];"
        f"[1:v]drawtext=text='{label_right}':fontcolor=white:fontsize=28:"
        f"x=10:y=h-th-10,format=yuv420p[right];"
        f"[left][right]hstack=inputs=2"
    )
    cmd = ['ffmpeg', '-y', '-i', left_mp4, '-i', right_mp4,
           '-filter_complex', filt, '-c:v', 'libx264', '-crf', '23',
           '-pix_fmt', 'yuv420p', output_mp4]
    logging.info(f"  composing comparison -> {output_mp4}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            logging.info(f"  -> {output_mp4}  ({os.path.getsize(output_mp4)/1e6:.1f} MB)")
            return True
        else:
            logging.error(f"  ffmpeg comparison failed: {r.stderr[-400:]}")
            return False
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Per-view pipeline (render + compose)
# ---------------------------------------------------------------------------

def process_view(data_dir, id_strs, view_name, pose_dir, video_frames_dir,
                 output_mp4, args, K, to_origin, bbox, fps):
    """Render (if needed) then compose one view."""
    need_render = args.mode in ('render', 'both')
    need_compose = args.mode in ('compose', 'both')

    if need_render:
        logging.info(f"-- Render: {view_name} --")
        color = (0, 255, 0) if view_name != 'fused' else (255, 128, 0)
        render_frames(data_dir, pose_dir, video_frames_dir, id_strs,
                      K, to_origin, bbox, left_color=color,
                      overwrite=args.overwrite)

    if need_compose:
        logging.info(f"-- Compose: {view_name} --")
        compose_video(video_frames_dir, output_mp4, id_strs, fps=fps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Build comparison videos from FP tracking results')
    parser.add_argument('--clip', type=str, default='clip03')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['render', 'compose', 'both'])
    parser.add_argument('--views', type=str, default='left,right,fused',
                        help='Comma-separated: left,right,fused (comparison auto-added if left+fused present)')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=-1)
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-render all video_frames (default: skip existing)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    data_dir = os.path.join(repo_dir, 'data', args.clip)
    mesh_file = os.path.join(data_dir, 'mesh', 'clean_mesh.obj')
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh not found: {mesh_file}")

    # Mesh scale
    scale_file = os.path.join(data_dir, 'foundationpose', 'run', 'scales', 'unified_scale.txt')
    mesh_scale = float(open(scale_file).read().strip()) if os.path.exists(scale_file) else 1.0
    logging.info(f"Mesh scale: {mesh_scale}")

    # Intrinsics
    K_path = os.path.join(data_dir, 'ffs', 'cam_K.txt')
    K = np.loadtxt(K_path).reshape(3, 3)
    logging.info(f"K:\n{K}")

    # Bounding box
    to_origin, bbox = compute_bbox(mesh_file, mesh_scale=mesh_scale)
    diag = np.linalg.norm(bbox[1] - bbox[0])
    logging.info(f"Bbox diagonal: {diag:.3f}m")

    # Frame list
    id_strs, _ = get_frame_list(data_dir, args.start_frame, args.end_frame)
    logging.info(f"Frames: {len(id_strs)}  (indices {args.start_frame}..{args.start_frame + len(id_strs) - 1})")

    views = [v.strip() for v in args.views.split(',')]
    vid_base = os.path.join(data_dir, 'foundationpose_v2')

    view_specs = {
        'left':  ('left',  os.path.join(vid_base, 'run', 'ob_in_cam'),
                  os.path.join(vid_base, 'run', 'video_frames'),
                  os.path.join(vid_base, 'run', 'track.mp4')),
        'right': ('right', os.path.join(vid_base, 'run_right', 'ob_in_cam'),
                  os.path.join(vid_base, 'run_right', 'video_frames'),
                  os.path.join(vid_base, 'run_right', 'track.mp4')),
        'fused': ('fused', os.path.join(vid_base, 'fused', 'ob_in_cam'),
                  os.path.join(vid_base, 'fused', 'video_frames'),
                  os.path.join(vid_base, 'fused', 'track.mp4')),
    }

    for v in views:
        if v not in view_specs:
            logging.warning(f"Unknown view '{v}', skipping. Options: {list(view_specs.keys())}")
            continue
        name, pose_dir, vf_dir, out_mp4 = view_specs[v]
        if not os.path.isdir(pose_dir):
            logging.warning(f"Pose directory not found: {pose_dir}, skip {name}")
            continue
        process_view(data_dir, id_strs, name, pose_dir, vf_dir, out_mp4,
                     args, K, to_origin, bbox, args.fps)

    # Side-by-side comparison: left vs fused
    if args.mode in ('compose', 'both'):
        left_mp4 = view_specs['left'][3]
        fused_mp4 = view_specs['fused'][3]
        comparison_mp4 = os.path.join(vid_base, 'comparison_left_fused.mp4')
        if os.path.exists(left_mp4) and os.path.exists(fused_mp4):
            logging.info("-- Comparison: left-only | fused --")
            compose_comparison(left_mp4, fused_mp4, comparison_mp4)
        else:
            logging.warning("Comparison skipped: need both left and fused videos")

    logging.info("Done.")


if __name__ == '__main__':
    main()
