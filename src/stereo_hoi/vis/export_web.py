"""Export HOI data as static web assets for the Three.js viewer.

Converts the Python Viser pipeline data into JSON / GLB / images that a
static HTML page can load directly — no backend server needed.
"""

import argparse
import json
import os
import shutil
import sys
import numpy as np

from .._pathresolver import paths
from ..hoi_data import (
    load_data, load_mano_faces, to_viser, BBOX_EDGES, bbox_corners_from_mesh,
)

_TV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_hand_display(hd: dict) -> list[dict]:
    """WiLoR hand data → viser-frame vertices for one frame."""
    vm = np.asarray(hd["verts_mano"], dtype=np.float32)
    jt = np.asarray(hd["joints"], dtype=np.float32)
    ir = np.asarray(hd["is_right"], dtype=np.uint8).reshape(-1)
    wr = (np.asarray(hd["wrist_3d"], dtype=np.float32)
          if hd.get("wrist_3d") is not None
          else np.zeros((len(vm), 3), dtype=np.float32))

    frame_hands = []
    for h_idx in range(min(len(vm), 2)):
        wrist_mano = jt[h_idx, 0]
        v_display = vm[h_idx] - wrist_mano + wr[h_idx]
        v_display = to_viser(v_display)
        frame_hands.append({
            "vertices": v_display.tolist(),
            "is_right": int(ir[h_idx]),
            "wrist": to_viser(wr[h_idx]).tolist(),
        })
    return frame_hands


def compute_bbox_segments(pose: np.ndarray,
                          bbox_h: np.ndarray) -> list[list[list[float]]]:
    """Viser-frame line segments for the bbox wireframe."""
    bbox_posed = (pose @ bbox_h.T).T[:, :3]
    bv = to_viser(bbox_posed)
    return [[bv[a].tolist(), bv[b].tolist()] for a, b in BBOX_EDGES]


def make_viser_pose(pose_cam: np.ndarray) -> list[list[float]]:
    """Convert FoundationPose 4x4 (OpenCV cam frame) → viser/Three.js frame."""
    return (_TV @ pose_cam).tolist()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(clip: str, *, step: int = 1, rgb: bool = False,
        out: str | None = None) -> None:
    """Export static web assets for a clip.

    Args:
        clip:  clip name.
        step:  frame subsampling step (1 = all frames).
        rgb:   also copy RGB thumbnails for 2D overlay.
        out:   output root (default: ``web_demo/static/results/<clip>``).
    """
    data_dir = str(paths.clip_dir(clip))
    out_dir = out or str(paths.web_demo_dir / "static" / "results" / clip)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading MANO faces...")
    mano_faces = load_mano_faces()

    print("Loading clip data...")
    data = load_data(data_dir)

    ids = data["pose_ids"][::step]
    N = len(ids)
    print(f"Exporting {N} frames (step={step})")

    obj_verts = data["obj_verts"]
    bbox_corners = data["obj_bbox_corners"]
    bbox_h = np.column_stack([bbox_corners, np.ones(8)])

    # 1. Object mesh as GLB
    print("Exporting object mesh...")
    import trimesh
    mesh_path = os.path.join(data_dir, "mesh", "clean_mesh.obj")
    mesh = trimesh.load(mesh_path)
    scale_path = os.path.join(data_dir, "foundationpose", "run",
                               "scales", "unified_scale.txt")
    if os.path.exists(scale_path):
        mesh.vertices *= float(open(scale_path).read().strip())
    glb_path = os.path.join(out_dir, "object_mesh.glb")
    mesh.export(glb_path)
    print(f"  -> {glb_path}")

    # 2. Per-frame object matrices + bbox
    print("Exporting object poses + bbox...")
    frames_data = {}
    for sid in ids:
        if sid not in data["poses"]:
            continue
        pose = data["poses"][sid]
        frames_data[sid] = {
            "matrix": make_viser_pose(pose),
            "bbox_segments": compute_bbox_segments(pose, bbox_h),
        }
    obj_path = os.path.join(out_dir, "object_frames.json")
    with open(obj_path, "w") as f:
        json.dump(frames_data, f)
    print(f"  -> {obj_path}  ({len(frames_data)} frames)")

    # 3. MANO faces
    faces_path = os.path.join(out_dir, "mano_faces.json")
    with open(faces_path, "w") as f:
        json.dump(mano_faces.tolist(), f)
    print(f"  -> {faces_path}")

    # 4. Hand meshes per frame
    print("Exporting hand meshes...")
    hand_frames = {}
    for sid in ids:
        hd = data["hands"].get(sid)
        if hd is None or hd.get("verts_mano") is None or hd.get("n_hands") == 0:
            continue
        hand_frames[sid] = compute_hand_display(hd)
    hands_path = os.path.join(out_dir, "hand_frames.json")
    with open(hands_path, "w") as f:
        json.dump(hand_frames, f)
    print(f"  -> {hands_path}  ({len(hand_frames)} frames with hands)")

    # 5. Metadata
    metadata = {
        "clip": clip, "step": step,
        "frames": ids, "num_frames": N, "fps": 30,
        "object_mesh": "object_mesh.glb",
        "object_frames": "object_frames.json",
        "hand_frames": "hand_frames.json",
        "mano_faces": "mano_faces.json",
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> {meta_path}")

    # 6. RGB thumbnails
    if rgb:
        import cv2
        rgb_src = os.path.join(data_dir, "rgb")
        rgb_dst = os.path.join(out_dir, "rgb")
        thumb_dst = os.path.join(out_dir, "thumbnails")
        os.makedirs(rgb_dst, exist_ok=True)
        os.makedirs(thumb_dst, exist_ok=True)
        n = 0
        has_rgb = []
        for sid in ids:
            for ext in (".jpg", ".png"):
                src = os.path.join(rgb_src, f"{sid}{ext}")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(rgb_dst, f"{sid}{ext}"))
                    img = cv2.imread(src)
                    h, w = img.shape[:2]
                    tw = 200
                    th = int(h * tw / w)
                    thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(thumb_dst, f"{sid}.jpg"), thumb,
                                [cv2.IMWRITE_JPEG_QUALITY, 75])
                    has_rgb.append(sid)
                    n += 1
                    break
        print(f"  -> {rgb_dst}/ + {thumb_dst}/  ({n} images)")

        with open(os.path.join(out_dir, "rgb_index.json"), "w") as f:
            json.dump({"has_rgb": has_rgb, "rgb_dir": "rgb",
                       "thumb_dir": "thumbnails"}, f)

    web_dir = paths.web_demo_dir
    print(f"\nDone.  Serve with:\n  cd {web_dir} && python -m http.server 8080")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HOI data for web demo")
    parser.add_argument("--clip", type=str, default="clip03")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    run(args.clip, step=args.step, rgb=args.rgb, out=args.out)


if __name__ == "__main__":
    main()
