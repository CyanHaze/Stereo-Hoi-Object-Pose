#!/usr/bin/env python
"""
Export HOI data as static web assets for the Three.js viewer.

Converts the Python Viser pipeline into JSON / GLB / images that a static
HTML page can load directly — no backend server needed.

Export strategy:
  - Object mesh  → single .glb, loaded once; per-frame 4×4 matrix applied in JS
  - Object bbox  → per-frame line segments (pre-computed in viser/Three.js frame)
  - Hand meshes  → per-frame vertex arrays (pre-computed in viser/Three.js frame)
  - MANO faces   → single .json (static topology)

Usage:
    conda activate diffusion
    cd F:/Research/02_Projects/SRTP/Reproduction

    python scripts/export_web_demo.py --clip clip03
    python scripts/export_web_demo.py --clip clip03 --step 3
    python scripts/export_web_demo.py --clip clip03 --rgb --step 5
"""

import argparse, os, sys, json, shutil
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(code_dir)
sys.path.insert(0, code_dir)

from hoi_data import load_data, load_mano_faces, to_viser, BBOX_EDGES, bbox_corners_from_mesh

# to_viser as a 4x4 matrix:  diag(1, -1, -1, 1)
_TV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_hand_display(hd):
    """WiLoR hand data → viser-frame vertices for one frame.

    Returns list of dicts: [{'vertices': (778,3), 'is_right': 0|1, 'wrist': (3,)}, ...]
    """
    vm = np.asarray(hd['verts_mano'], dtype=np.float32)
    jt = np.asarray(hd['joints'], dtype=np.float32)
    ir = np.asarray(hd['is_right'], dtype=np.uint8).reshape(-1)
    wr = np.asarray(hd['wrist_3d'], dtype=np.float32) if hd.get('wrist_3d') is not None \
         else np.zeros((len(vm), 3), dtype=np.float32)

    frame_hands = []
    for h_idx in range(min(len(vm), 2)):
        wrist_mano = jt[h_idx, 0]
        v_display = vm[h_idx] - wrist_mano + wr[h_idx]   # metric cam frame
        v_display = to_viser(v_display)                    # viser / Three.js frame
        frame_hands.append({
            'vertices': v_display.tolist(),
            'is_right': int(ir[h_idx]),
            'wrist': to_viser(wr[h_idx]).tolist(),
        })
    return frame_hands


def compute_bbox_segments(pose, bbox_h):
    """Return viser-frame line segments for the bbox wireframe."""
    bbox_posed = (pose @ bbox_h.T).T[:, :3]
    bv = to_viser(bbox_posed)
    segs = [[bv[a].tolist(), bv[b].tolist()] for a, b in BBOX_EDGES]
    return segs


def make_viser_pose(pose_cam):
    """Convert 4x4 FoundationPose matrix (OpenCV cam frame) → viser / Three.js frame."""
    return (_TV @ pose_cam).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Export HOI data for web demo')
    parser.add_argument('--clip', type=str, default='clip03')
    parser.add_argument('--step', type=int, default=1,
                        help='Frame subsampling step (default 1 = all frames)')
    parser.add_argument('--rgb', action='store_true',
                        help='Also copy RGB thumbnails for 2D overlay')
    parser.add_argument('--out', type=str, default=None,
                        help='Output root (default: repo/web_demo/static/results/<clip>)')
    args = parser.parse_args()

    data_dir = os.path.join(repo_dir, 'data', args.clip)
    out_dir = args.out or os.path.join(repo_dir, 'web_demo', 'static', 'results', args.clip)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load ----
    print("Loading MANO faces...")
    mano_faces = load_mano_faces()

    print("Loading clip data...")
    data = load_data(data_dir)

    ids = data['pose_ids'][::args.step]
    N = len(ids)
    print(f"Exporting {N} frames (step={args.step})")

    obj_verts = data['obj_verts']
    bbox_corners = data['obj_bbox_corners']
    bbox_h = np.column_stack([bbox_corners, np.ones(8)])

    # ---- 1. Object mesh as GLB (static, loaded once) ----
    print("Exporting object mesh...")
    import trimesh
    mesh_path = os.path.join(data_dir, 'mesh', 'clean_mesh.obj')
    mesh = trimesh.load(mesh_path)
    scale_file = os.path.join(data_dir, 'foundationpose', 'run', 'scales',
                              'unified_scale.txt')
    if os.path.exists(scale_file):
        mesh.vertices *= float(open(scale_file).read().strip())
    glb_path = os.path.join(out_dir, 'object_mesh.glb')
    mesh.export(glb_path)
    print(f"  -> {glb_path}")

    # ---- 2. Per-frame data: object 4x4 matrix + bbox segments ----
    print("Exporting object poses + bbox...")
    frames_data = {}
    for sid in ids:
        if sid not in data['poses']:
            continue
        pose = data['poses'][sid]
        frames_data[sid] = {
            'matrix': make_viser_pose(pose),
            'bbox_segments': compute_bbox_segments(pose, bbox_h),
        }
    obj_path = os.path.join(out_dir, 'object_frames.json')
    with open(obj_path, 'w') as f:
        json.dump(frames_data, f)
    print(f"  -> {obj_path}  ({len(frames_data)} frames)")

    # ---- 3. MANO faces (static) ----
    faces_path = os.path.join(out_dir, 'mano_faces.json')
    with open(faces_path, 'w') as f:
        json.dump(mano_faces.tolist(), f)
    print(f"  -> {faces_path}")

    # ---- 4. Hand meshes (per-frame, pre-computed viser-frame vertices) ----
    print("Exporting hand meshes...")
    hand_frames = {}
    for sid in ids:
        hd = data['hands'].get(sid)
        if hd is None or hd.get('verts_mano') is None or hd.get('n_hands') == 0:
            continue
        hand_frames[sid] = compute_hand_display(hd)
    hands_path = os.path.join(out_dir, 'hand_frames.json')
    with open(hands_path, 'w') as f:
        json.dump(hand_frames, f)
    print(f"  -> {hands_path}  ({len(hand_frames)} frames with hands)")

    # ---- 5. Metadata ----
    metadata = {
        'clip': args.clip,
        'step': args.step,
        'frames': ids,
        'num_frames': N,
        'fps': 30,
        'object_mesh': 'object_mesh.glb',
        'object_frames': 'object_frames.json',
        'hand_frames': 'hand_frames.json',
        'mano_faces': 'mano_faces.json',
    }
    meta_path = os.path.join(out_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> {meta_path}")

    # ---- 6. RGB thumbnails ----
    # Full-size frames (for 3D overlay) + small thumbnails (for frame strip)
    if args.rgb:
        import cv2
        rgb_src = os.path.join(data_dir, 'rgb')
        rgb_dst = os.path.join(out_dir, 'rgb')
        thumb_dst = os.path.join(out_dir, 'thumbnails')
        os.makedirs(rgb_dst, exist_ok=True)
        os.makedirs(thumb_dst, exist_ok=True)
        n = 0
        has_rgb = []
        for sid in ids:
            for ext in ('.jpg', '.png'):
                src = os.path.join(rgb_src, f'{sid}{ext}')
                if os.path.exists(src):
                    # Full-size copy
                    shutil.copy2(src, os.path.join(rgb_dst, f'{sid}{ext}'))
                    # Thumbnail (200px wide, JPEG quality 75)
                    img = cv2.imread(src)
                    h, w = img.shape[:2]
                    tw = 200
                    th = int(h * tw / w)
                    thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(thumb_dst, f'{sid}.jpg'), thumb,
                                [cv2.IMWRITE_JPEG_QUALITY, 75])
                    has_rgb.append(sid)
                    n += 1
                    break
        print(f"  -> {rgb_dst}/ + {thumb_dst}/  ({n} images)")

        # Save frame → thumbnail mapping so the viewer knows which frames have RGB
        with open(os.path.join(out_dir, 'rgb_index.json'), 'w') as f:
            json.dump({'has_rgb': has_rgb, 'rgb_dir': 'rgb', 'thumb_dir': 'thumbnails'}, f)

    print(f"\nDone.  Serve with:\n"
          f"  cd {os.path.join(repo_dir, 'web_demo')} && python -m http.server 8080")


if __name__ == '__main__':
    main()
