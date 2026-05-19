#!/usr/bin/env python
"""
Shared data loading and coordinate utilities for HOI visualisation.

Used by: hoi_viewer.py (local Viser), export_web_demo.py (web export),
         render_hoi.py (2D overlay).
"""

import os, sys, glob, json
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(code_dir)
WILOR_DIR = os.path.join(repo_dir, 'WiLoR')

# ---------------------------------------------------------------------------
# BBox wireframe edges
# ---------------------------------------------------------------------------

BBOX_EDGES = np.array([
    [0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4], [0,4],[1,5],[2,6],[3,7],
], dtype=np.int32)


def bbox_corners_from_mesh(verts):
    """8 corners of axis-aligned bounding box."""
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    return np.array([
        [mn[0],mn[1],mn[2]], [mx[0],mn[1],mn[2]], [mx[0],mx[1],mn[2]], [mn[0],mx[1],mn[2]],
        [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]], [mx[0],mx[1],mx[2]], [mn[0],mx[1],mx[2]],
    ])


# ---------------------------------------------------------------------------
# MANO faces
# ---------------------------------------------------------------------------

def load_mano_faces():
    """Return MANO hand faces (1538, 3).

    Loads from a cached .npy file alongside the WiLoR checkpoint.  If the cache
    doesn't exist yet, we import WiLoR once (slow) to extract the faces, then
    save them so subsequent launches are instant.
    """
    cache_path = os.path.join(WILOR_DIR, 'pretrained_models', 'mano_faces.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path).astype(np.int32)

    print("Extracting MANO faces from WiLoR checkpoint (one-time, ~10 s)...")
    import torch
    _orig = torch.load
    def _patched(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig(*a, **kw)
    torch.load = _patched

    cwd = os.getcwd()
    os.chdir(WILOR_DIR)
    sys.path.insert(0, '.')
    try:
        from wilor.models import load_wilor
        model, _ = load_wilor(
            './pretrained_models/wilor_final.ckpt',
            './pretrained_models/model_config.yaml',
        )
        faces = model.mano.faces.astype(np.int32)
        np.save(cache_path, faces)
        print(f"  -> cached to {cache_path}")
        return faces
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def to_viser(pts):
    """Convert camera-frame (X right, Y down, Z forward) → viser (X right, Y up, Z back)."""
    p = np.asarray(pts, dtype=np.float32).copy()
    if p.ndim == 1:
        p[1] *= -1; p[2] *= -1
    else:
        p[:, 1] *= -1; p[:, 2] *= -1
    return p


# ---------------------------------------------------------------------------
# Clip data loader
# ---------------------------------------------------------------------------

def load_data(data_dir):
    """Load all required data for a clip.  Returns dict with keys:
        obj_verts, obj_faces, obj_bbox_corners  — object mesh
        pose_ids, poses                         — FoundationPose results
        hand_ids, hands                         — WiLoR hand data
        common_ids                              — frames with both
        calib                                   — camera calibration
    """
    data = {}

    # ---- object mesh ----
    mesh_path = os.path.join(data_dir, 'mesh', 'clean_mesh.obj')
    import trimesh
    mesh = trimesh.load(mesh_path)
    verts_obj = mesh.vertices.copy()
    faces_obj = mesh.faces.copy()

    scale_file = os.path.join(data_dir, 'foundationpose', 'run', 'scales', 'unified_scale.txt')
    if os.path.exists(scale_file):
        scale = float(open(scale_file).read().strip())
        verts_obj *= scale
        print(f"Mesh scale: {scale}")

    data['obj_verts'] = verts_obj.astype(np.float32)
    data['obj_faces'] = faces_obj.astype(np.int32)
    data['obj_bbox_corners'] = bbox_corners_from_mesh(verts_obj).astype(np.float32)

    # ---- object poses (fused) ----
    pose_dir = os.path.join(data_dir, 'foundationpose_v2', 'fused', 'ob_in_cam')
    pose_files = sorted(glob.glob(os.path.join(pose_dir, '*.txt')))
    data['pose_ids'] = [os.path.splitext(os.path.basename(f))[0] for f in pose_files]
    data['poses'] = {}
    for pid, pf in zip(data['pose_ids'], pose_files):
        data['poses'][pid] = np.loadtxt(pf).reshape(4, 4).astype(np.float32)
    print(f"Object poses: {len(data['poses'])} frames")

    # ---- hand data ----
    hand_dir = os.path.join(data_dir, 'wilor', 'left')
    hand_files = sorted(glob.glob(os.path.join(hand_dir, '*.npz')))
    data['hand_ids'] = [os.path.splitext(os.path.basename(f))[0] for f in hand_files]
    data['hands'] = {}
    for hid, hf in zip(data['hand_ids'], hand_files):
        d = dict(np.load(hf, allow_pickle=True))
        data['hands'][hid] = {
            'verts_mano': d.get('verts_mano', None),
            'joints': d.get('joints', None),
            'is_right': d.get('is_right', None),
            'wrist_3d': d.get('wrist_3d', None),
            'depth_ok': d.get('depth_ok', None),
            'n_hands': d.get('n_hands', None),
        }
    print(f"Hand data: {len(data['hands'])} frames")

    # ---- common frames ----
    common = sorted(set(data['pose_ids']) & set(data['hand_ids']))
    data['common_ids'] = common
    print(f"Frames with hand data: {len(common)}")

    # ---- calibration ----
    with open(os.path.join(data_dir, 'calib.json')) as f:
        data['calib'] = json.load(f)

    return data
