"""Combined hand + object 2D overlay rendering.

Overlays the MANO hand mesh and the object mesh (with oriented bounding box)
on RGB frames, using FFS-depth-aligned metric 3D coordinates.

Requires: WiLoR inference results + FoundationPose tracking results on disk.
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
import time
import cv2
import numpy as np

from .._pathresolver import paths
from ..hand.alignment import render_mesh_overlay


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def load_obj_verts_faces(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and faces from .obj (quads → triangles)."""
    verts = []
    faces = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith("f "):
                parts = [p.split("/")[0] for p in line.strip().split()[1:]]
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

def compute_oriented_bbox(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oriented bounding box of a point cloud.

    Returns ``(to_origin_4x4, bbox_2x3)`` where *bbox* rows are min/max
    corners in the oriented frame.
    """
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
# 3D bbox wireframe drawing
# ---------------------------------------------------------------------------

def _to_homo(pts: np.ndarray) -> np.ndarray:
    return np.concatenate((pts, np.ones((len(pts), 1))), axis=-1)


def draw_posed_3d_box(K: np.ndarray, img: np.ndarray,
                      ob_in_cam: np.ndarray, bbox: np.ndarray,
                      line_color: tuple = (0, 255, 0),
                      linewidth: int = 2) -> np.ndarray:
    """Draw a 3D bounding box wireframe projected onto *img* (in-place)."""
    xmin, ymin, zmin = bbox.min(axis=0)
    xmax, ymax, zmax = bbox.max(axis=0)

    def _line(start, end):
        pts = np.stack((start, end)).reshape(-1, 3)
        pts = (ob_in_cam @ _to_homo(pts).T).T[:, :3]
        proj = (K @ pts.T).T
        uv = np.round(proj[:, :2] / proj[:, 2].reshape(-1, 1)).astype(int)
        cv2.line(img, tuple(uv[0].tolist()), tuple(uv[1].tolist()),
                 color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)

    for y in (ymin, ymax):
        for z in (zmin, zmax):
            _line(np.array([xmin, y, z]), np.array([xmax, y, z]))
    for x in (xmin, xmax):
        for z in (zmin, zmax):
            _line(np.array([x, ymin, z]), np.array([x, ymax, z]))
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            _line(np.array([x, y, zmin]), np.array([x, y, zmax]))
    return img


# ---------------------------------------------------------------------------
# ffmpeg MP4 composer
# ---------------------------------------------------------------------------

def compose_video_ffmpeg(frame_dir: str, output_mp4: str, fps: int) -> None:
    """Compose PNG frames to MP4 via ffmpeg concat demuxer."""
    files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not files:
        return
    list_path = output_mp4.replace(".mp4", "_concat.txt")
    with open(list_path, "w") as fl:
        for p in files:
            fl.write(f"file '{p.replace(chr(92), '/')}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-r", str(fps),
        "-i", list_path, "-c:v", "libx264", "-crf", "23",
        "-pix_fmt", "yuv420p", output_mp4,
    ]
    logging.info("Composing %d frames → %s", len(files), output_mp4)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        os.remove(list_path)
        logging.info("  → %s (%.1f MB)", output_mp4,
                     os.path.getsize(output_mp4) / 1e6)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(clip: str, camera: str = "left", *,
        start_frame: int = 0,
        end_frame: int = -1,
        fps: int = 0,
        overwrite: bool = False,
        ) -> None:
    """Render hand + object overlay frames for a clip.

    Args:
        clip:       clip name.
        camera:     ``'left'`` (uses fused pose) or ``'right'``.
        start_frame: first frame index.
        end_frame:   last frame index (exclusive; -1 = all).
        fps:         if > 0, also compose MP4 after rendering.
        overwrite:   re-render existing frames.
    """
    data_dir = str(paths.clip_dir(clip))
    hand_dir = os.path.join(data_dir, "wilor", camera)
    obj_pose_dir = os.path.join(
        data_dir, "foundationpose_v2",
        "fused" if camera == "left" else "run_right",
        "ob_in_cam",
    )
    rgb_dir = os.path.join(data_dir, "rgb" if camera == "left" else "right")
    out_dir = os.path.join(data_dir, "hoi", "video_frames")
    os.makedirs(out_dir, exist_ok=True)

    K = np.loadtxt(os.path.join(data_dir, "ffs", "cam_K.txt")).reshape(3, 3)

    # Object mesh
    mesh_file = os.path.join(data_dir, "mesh", "clean_mesh.obj")
    obj_verts, obj_faces = load_obj_verts_faces(mesh_file)
    scale_path = os.path.join(data_dir, "foundationpose", "run",
                               "scales", "unified_scale.txt")
    mesh_scale = float(open(scale_path).read().strip()) if os.path.exists(scale_path) else 1.0
    obj_verts *= mesh_scale
    to_origin, bbox = compute_oriented_bbox(obj_verts)
    logging.info("Object mesh: %dv, %df, scale=%.4f, bbox diag=%.3f m",
                 len(obj_verts), len(obj_faces), mesh_scale,
                 np.linalg.norm(bbox[1] - bbox[0]))

    # MANO faces
    mano_faces_path = os.path.join(data_dir, "mano_faces.npy")
    if os.path.exists(mano_faces_path):
        mano_faces = np.load(mano_faces_path)
    else:
        sys.path.insert(0, str(paths.wilor_dir))
        _cwd = os.getcwd()
        os.chdir(str(paths.wilor_dir))
        try:
            import torch
            _orig = torch.load
            def _p(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig(*a, **kw)
            torch.load = _p
            from wilor.models import load_wilor
            m, _ = load_wilor("./pretrained_models/wilor_final.ckpt",
                              "./pretrained_models/model_config.yaml")
            mano_faces = m.mano.faces
            np.save(mano_faces_path, mano_faces)
        finally:
            os.chdir(_cwd)

    hand_files = sorted(glob.glob(os.path.join(hand_dir, "*.npz")))
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in hand_files]
    total = len(id_strs)
    _end = total if end_frame <= 0 else min(end_frame, total)
    id_strs = id_strs[start_frame:_end]

    todo = sum(1 for sid in id_strs
               if overwrite
               or not os.path.exists(os.path.join(out_dir, f"{sid}.png")))

    if todo == 0:
        logging.info("All frames already rendered.")
        if fps > 0:
            compose_video_ffmpeg(out_dir, os.path.join(data_dir, "hoi", "track.mp4"), fps)
        return

    logging.info("Rendering %d frames → %s/", todo, out_dir)
    t0 = time.time()

    for n, sid in enumerate(id_strs):
        out_path = os.path.join(out_dir, f"{sid}.png")
        if os.path.exists(out_path) and not overwrite:
            continue

        img_path = os.path.join(rgb_dir, f"{sid}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(rgb_dir, f"{sid}.png")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Render object
        obj_pose_path = os.path.join(obj_pose_dir, f"{sid}.txt")
        if os.path.exists(obj_pose_path):
            pose = np.loadtxt(obj_pose_path).reshape(4, 4)
            obj_verts_cam = (pose[:3, :3] @ obj_verts.T).T + pose[:3, 3]
            img_bgr = render_mesh_overlay(
                img_bgr, obj_verts_cam, obj_faces, K,
                color=(100, 200, 80), alpha=0.45,
            )
            center_pose = pose @ np.linalg.inv(to_origin)
            draw_posed_3d_box(K, img_bgr, center_pose, bbox,
                              line_color=(0, 220, 80), linewidth=2)

        # Render hands
        hand_path = os.path.join(hand_dir, f"{sid}.npz")
        if os.path.exists(hand_path):
            hd = np.load(hand_path)
            N = len(hd["verts_virt"])
            for hn in range(N):
                is_r = hd["is_right"][hn]
                color = (255, 128, 0) if is_r else (0, 220, 220)
                w = hd["wrist_3d"][hn]
                ok = hd["depth_ok"][hn]
                wrist_mano = hd["joints"][hn, 0].astype(np.float32)
                verts_hand_cam = hd["verts_mano"][hn] - wrist_mano + w
                img_bgr = render_mesh_overlay(
                    img_bgr, verts_hand_cam, mano_faces, K,
                    color=color, alpha=0.45,
                )
                if ok and np.linalg.norm(w) > 0.001:
                    pw = K @ w
                    pw = (pw[:2] / pw[2]).astype(int)
                    cv2.circle(img_bgr, tuple(pw.tolist()), 5,
                               (0, 0, 255), -1, lineType=cv2.LINE_AA)

        cv2.imwrite(out_path, img_bgr)

        if (n + 1) % 50 == 0:
            e = time.time() - t0
            logging.info("  %d/%d  (%.0f s, %.1f fps)",
                         n + 1, todo, e, (n + 1) / e)

    e = time.time() - t0
    logging.info("Done: %d frames in %.0f s → %s/", todo, e, out_dir)

    if fps > 0:
        compose_video_ffmpeg(out_dir, os.path.join(data_dir, "hoi", "track.mp4"), fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="HOI combined rendering")
    parser.add_argument("--clip", type=str, default="clip03")
    parser.add_argument("--camera", type=str, default="left")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(args.clip, args.camera,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
