"""Shared data loading and coordinate utilities for HOI visualisation.

Used by: ``vis.viewer``, ``vis.export_web``, ``vis.hoi_render``.
"""

import glob
import json
import os
import sys
import numpy as np

from ._pathresolver import paths

# ---------------------------------------------------------------------------
# BBox wireframe edges
# ---------------------------------------------------------------------------

BBOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)


def bbox_corners_from_mesh(verts: np.ndarray) -> np.ndarray:
    """8 corners of axis-aligned bounding box."""
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    return np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])


# ---------------------------------------------------------------------------
# MANO faces (cached)
# ---------------------------------------------------------------------------

def load_mano_faces() -> np.ndarray:
    """Return MANO hand faces (1538, 3).

    Loads from a cached ``.npy`` in the WiLoR pretrained_models directory.
    If the cache doesn't exist, import WiLoR once to extract the faces.
    """
    cache_path = paths.wilor_dir / "pretrained_models" / "mano_faces.npy"
    if cache_path.exists():
        return np.load(str(cache_path)).astype(np.int32)

    print("Extracting MANO faces from WiLoR checkpoint (one-time, ~10 s)...")
    import torch
    _orig = torch.load

    def _patched(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _patched

    cwd = os.getcwd()
    os.chdir(str(paths.wilor_dir))
    sys.path.insert(0, ".")
    try:
        from wilor.models import load_wilor
        model, _ = load_wilor(
            "./pretrained_models/wilor_final.ckpt",
            "./pretrained_models/model_config.yaml",
        )
        faces = model.mano.faces.astype(np.int32)
        np.save(str(cache_path), faces)
        print(f"  -> cached to {cache_path}")
        return faces
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def to_viser(pts: np.ndarray) -> np.ndarray:
    """Camera frame (X right, Y down, Z forward) → viser (X right, Y up, Z back)."""
    p = np.asarray(pts, dtype=np.float32).copy()
    if p.ndim == 1:
        p[1] *= -1
        p[2] *= -1
    else:
        p[:, 1] *= -1
        p[:, 2] *= -1
    return p


# ---------------------------------------------------------------------------
# Clip data loader
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> dict:
    """Load all required data for a clip.

    Returns a dict with keys:
        obj_verts, obj_faces, obj_bbox_corners  — object mesh
        pose_ids, poses                         — FoundationPose results
        hand_ids, hands                         — WiLoR hand data
        common_ids                              — frames with both
        calib                                   — camera calibration
    """
    data: dict = {}

    # Object mesh
    mesh_path = os.path.join(data_dir, "mesh", "clean_mesh.obj")
    import trimesh
    mesh = trimesh.load(mesh_path)
    verts_obj = mesh.vertices.copy()
    faces_obj = mesh.faces.copy()

    scale_file = os.path.join(data_dir, "foundationpose", "run",
                               "scales", "unified_scale.txt")
    if os.path.exists(scale_file):
        scale = float(open(scale_file).read().strip())
        verts_obj *= scale
        print(f"Mesh scale: {scale}")

    data["obj_verts"] = verts_obj.astype(np.float32)
    data["obj_faces"] = faces_obj.astype(np.int32)
    data["obj_bbox_corners"] = bbox_corners_from_mesh(verts_obj).astype(np.float32)

    # Object poses (fused)
    pose_dir = os.path.join(data_dir, "foundationpose_v2", "fused", "ob_in_cam")
    pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    data["pose_ids"] = [os.path.splitext(os.path.basename(f))[0]
                        for f in pose_files]
    data["poses"] = {}
    for pid, pf in zip(data["pose_ids"], pose_files):
        data["poses"][pid] = np.loadtxt(pf).reshape(4, 4).astype(np.float32)
    print(f"Object poses: {len(data['poses'])} frames")

    # Hand data
    hand_dir = os.path.join(data_dir, "wilor", "left")
    hand_files = sorted(glob.glob(os.path.join(hand_dir, "*.npz")))
    data["hand_ids"] = [os.path.splitext(os.path.basename(f))[0]
                        for f in hand_files]
    data["hands"] = {}
    for hid, hf in zip(data["hand_ids"], hand_files):
        d = dict(np.load(hf, allow_pickle=True))
        data["hands"][hid] = {
            "verts_mano": d.get("verts_mano", None),
            "joints": d.get("joints", None),
            "is_right": d.get("is_right", None),
            "wrist_3d": d.get("wrist_3d", None),
            "depth_ok": d.get("depth_ok", None),
            "n_hands": d.get("n_hands", None),
        }
    print(f"Hand data: {len(data['hands'])} frames")

    # Common frames
    common = sorted(set(data["pose_ids"]) & set(data["hand_ids"]))
    data["common_ids"] = common
    print(f"Frames with hand data: {len(common)}")

    # Calibration
    with open(os.path.join(data_dir, "calib.json")) as f:
        data["calib"] = json.load(f)

    return data
