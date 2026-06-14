"""Build comparison videos from FoundationPose tracking results.

Two modes:
  render  — generate video_frames from ob_in_cam/*.txt poses
  compose — ffmpeg video_frames/*.png → MP4
  both    — render + compose (default)

Output videos (placed alongside video_frames/):
  foundationpose_v2/run/track.mp4           (left-only)
  foundationpose_v2/run_right/track.mp4     (right-only)
  foundationpose_v2/fused/track.mp4         (fused)
  foundationpose_v2/comparison_left_fused.mp4  (side-by-side)
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import cv2
import numpy as np

from .._pathresolver import paths


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

def _compose_mp4(frame_dir: str, output_mp4: str, fps: int) -> None:
    files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not files:
        logging.warning("No frames in %s", frame_dir)
        return
    list_path = output_mp4.replace(".mp4", "_concat.txt")
    with open(list_path, "w") as fh:
        for p in files:
            fh.write(f"file '{p.replace(chr(92), '/')}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-r", str(fps), "-i", list_path,
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
        output_mp4,
    ]
    logging.info("Composing %d frames → %s", len(files), output_mp4)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        os.remove(list_path)
        logging.info("  → %s (%.1f MB)", output_mp4,
                     os.path.getsize(output_mp4) / 1e6)


def _compose_side_by_side(left_dir: str, right_dir: str,
                          output_mp4: str, fps: int, ids: list[str]) -> None:
    import tempfile
    left_files = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_files = sorted(glob.glob(os.path.join(right_dir, "*.png")))

    with tempfile.TemporaryDirectory() as tmp:
        for i, sid in enumerate(ids):
            if i >= len(left_files) or i >= len(right_files):
                break
            left = cv2.imread(left_files[i])
            right = cv2.imread(right_files[i])
            if left is None or right is None:
                continue
            # Labels
            cv2.putText(left, "left-only", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(right, "fused", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            side = np.hstack([left, right])
            cv2.imwrite(os.path.join(tmp, f"{sid}.png"), side)

        _compose_mp4(tmp, output_mp4, fps)


# ---------------------------------------------------------------------------
# Rendered frame generation (pure numpy + cv2, no FP import)
# ---------------------------------------------------------------------------

def _load_bbox(data_dir: str, mesh_file: str):
    """Return (to_origin_4x4, bbox_2x3)."""
    try:
        import trimesh
        mesh = trimesh.load(mesh_file)
        scale_path = os.path.join(data_dir, "foundationpose", "run",
                                   "scales", "unified_scale.txt")
        if os.path.exists(scale_path):
            mesh.vertices *= float(open(scale_path).read().strip())
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    except ImportError:
        obj_verts, _ = _load_obj_verts(mesh_file)
        center = (obj_verts.min(axis=0) + obj_verts.max(axis=0)) / 2
        extents = obj_verts.max(axis=0) - obj_verts.min(axis=0)
        to_origin = np.eye(4)
        to_origin[:3, 3] = center
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    return to_origin, bbox


def _load_obj_verts(mesh_file: str):
    verts = []
    with open(mesh_file) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.strip().split()[1:4]])
    return np.array(verts, dtype=np.float32)


def _draw_bbox(K, img, ob_in_cam, bbox,
               color=(0, 255, 0), lw=2):
    """Draw 3D bbox wireframe on img (in-place)."""
    xmin, ymin, zmin = bbox.min(axis=0)
    xmax, ymax, zmax = bbox.max(axis=0)

    def _line(a, b_):
        pts = np.stack([a, b_]).reshape(-1, 3)
        pts = (ob_in_cam @ np.column_stack([pts, np.ones(2)]).T).T[:, :3]
        uv = (K @ pts.T).T
        uv = np.round(uv[:, :2] / uv[:, 2:3]).astype(int)
        cv2.line(img, tuple(uv[0]), tuple(uv[1]),
                 color=color, thickness=lw, lineType=cv2.LINE_AA)

    for y in (ymin, ymax):
        for z in (zmin, zmax):
            _line([xmin, y, z], [xmax, y, z])
    for x in (xmin, xmax):
        for z in (zmin, zmax):
            _line([x, ymin, z], [x, ymax, z])
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            _line([x, y, zmin], [x, y, zmax])


def _render_video_frames(data_dir: str, run_dir: str, mesh_file: str,
                         start: int, end: int,
                         H: int, W: int, K: np.ndarray) -> None:
    """Generate video_frames/*.png from ob_in_cam/*.txt (no FP import)."""
    pose_dir = os.path.join(run_dir, "ob_in_cam")
    out_dir = os.path.join(run_dir, "video_frames")
    os.makedirs(out_dir, exist_ok=True)

    to_origin, bbox = _load_bbox(data_dir, mesh_file)

    pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in pose_files]

    rgb_dir = os.path.join(data_dir, "rgb")
    _end = min(end, len(pose_files))

    for i in range(start, _end):
        sid = id_strs[i]
        out_path = os.path.join(out_dir, f"{sid}.png")
        if os.path.exists(out_path):
            continue

        pose = np.loadtxt(pose_files[i]).reshape(4, 4)
        center_pose = pose @ np.linalg.inv(to_origin)

        # RGB
        rgb_path = os.path.join(rgb_dir, f"{sid}.jpg")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(rgb_dir, f"{sid}.png")
        color = cv2.imread(rgb_path)
        if color is None:
            continue
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)

        _draw_bbox(K, color, center_pose, bbox)
        cv2.imwrite(out_path, color)

        if (i + 1) % 50 == 0:
            logging.info("  [render] %d/%d", i + 1, _end)

    logging.info("  [render] Done: %d frames → %s/", _end - start, out_dir)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(clip: str, *,
        mode: str = "both",
        views: str = "left,fused",
        fps: int = 15,
        start_frame: int = 0,
        end_frame: int = -1,
        ) -> None:
    """Build comparison videos for a clip.

    Args:
        clip:  clip name.
        mode:  ``'render'``, ``'compose'``, or ``'both'``.
        views: comma-separated list: ``'left'``, ``'right'``, ``'fused'``.
        fps:   output video frame rate.
        start_frame / end_frame: frame range.
    """
    data_dir = str(paths.clip_dir(clip))
    view_list = [v.strip() for v in views.split(",")]

    K_full = np.loadtxt(os.path.join(data_dir, "ffs", "cam_K.txt")).reshape(3, 3)
    sample = cv2.imread(os.path.join(data_dir, "rgb", "00000.jpg"))
    if sample is None:
        sample = cv2.imread(os.path.join(data_dir, "rgb", "00000.png"))
    H_orig, W_orig = sample.shape[:2]
    shorter = 800
    scale = shorter / min(H_orig, W_orig)
    H, W = int(H_orig * scale), int(W_orig * scale)
    K = K_full.copy()
    K[:2] *= scale

    mesh_file = os.path.join(data_dir, "mesh", "clean_mesh.obj")

    run_map = {
        "left": os.path.join(data_dir, "foundationpose_v2", "run"),
        "right": os.path.join(data_dir, "foundationpose_v2", "run_right"),
        "fused": os.path.join(data_dir, "foundationpose_v2", "fused"),
    }

    if mode in ("render", "both"):
        for v in view_list:
            rdir = run_map.get(v)
            if rdir and os.path.isdir(os.path.join(rdir, "ob_in_cam")):
                logging.info("Rendering %s video_frames...", v)
                _render_video_frames(data_dir, rdir, mesh_file,
                                     start_frame, end_frame, H, W, K)

    if mode in ("compose", "both"):
        for v in view_list:
            rdir = run_map.get(v)
            if rdir:
                out_mp4 = os.path.join(rdir, "track.mp4")
                vf_dir = os.path.join(rdir, "video_frames")
                if os.path.isdir(vf_dir):
                    _compose_mp4(vf_dir, out_mp4, fps)

        # Side-by-side comparison: left-only | fused
        if "left" in view_list and "fused" in view_list:
            left_vf = os.path.join(run_map["left"], "video_frames")
            fused_vf = os.path.join(run_map["fused"], "video_frames")
            if os.path.isdir(left_vf) and os.path.isdir(fused_vf):
                # Build frame ID list
                pose_files = sorted(glob.glob(
                    os.path.join(run_map["left"], "ob_in_cam", "*.txt")))
                ids = [os.path.splitext(os.path.basename(f))[0]
                       for f in pose_files]
                out_mp4 = os.path.join(data_dir, "foundationpose_v2",
                                       "comparison_left_fused.mp4")
                _compose_side_by_side(left_vf, fused_vf, out_mp4, fps, ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build comparison videos from FP tracking results")
    parser.add_argument("--clip", type=str, default="clip03")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["render", "compose", "both"])
    parser.add_argument("--views", type=str, default="left,fused")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    args = parser.parse_args()

    run(args.clip, mode=args.mode, views=args.views, fps=args.fps,
        start_frame=args.start_frame, end_frame=args.end_frame)


if __name__ == "__main__":
    main()
