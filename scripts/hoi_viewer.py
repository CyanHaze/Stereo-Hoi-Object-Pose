#!/usr/bin/env python
"""
Interactive 3D viewer for hand-object interaction using Viser.

Renders object mesh (at estimated pose) + hand meshes (WiLoR metric coords)
in a shared 3D scene with free orbit/pan/zoom camera, time slider, play/pause.

Camera convention: OpenCV (X right, Y down, Z forward) → viser (X right, Y up, Z back).
All 3D quantities are in metric meters.

Usage:
    conda activate diffusion
    cd F:/Research/02_Projects/SRTP/Reproduction
    python scripts/hoi_viewer.py --clip clip03
    python scripts/hoi_viewer.py --clip clip03 --host 127.0.0.1 --port 8080 --fps 15
"""

import argparse, os, sys, glob, json, logging, time, threading
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(code_dir)
WILOR_DIR = os.path.join(repo_dir, 'WiLoR')
sys.path.insert(0, WILOR_DIR)


# ---------------------------------------------------------------------------
# MANO faces (loaded once from WiLoR model)
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
# BBox wireframe helper
# ---------------------------------------------------------------------------

_BBOX_EDGES = np.array([
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
# Data loader
# ---------------------------------------------------------------------------

def load_data(data_dir):
    """Load all required data for the clip.  Returns dict."""
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
        # convert npz arrays to proper format
        data['hands'][hid] = {
            'verts_mano': d.get('verts_mano', None),   # full 3D shape (~14cm Z range)
            'joints': d.get('joints', None),            # 3D joints, [0]=wrist
            'is_right': d.get('is_right', None),
            'wrist_3d': d.get('wrist_3d', None),        # metric wrist (camera frame)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='HOI 3D viewer')
    parser.add_argument('--clip', type=str, default='clip03')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    import viser

    data_dir = os.path.join(repo_dir, 'data', args.clip)

    # ---- load assets ----
    print("Loading MANO faces...")
    mano_faces = load_mano_faces()

    print("Loading clip data...")
    data = load_data(data_dir)
    # Timeline = ALL object pose frames; hand data may be sparse
    ids = data['pose_ids']
    N = len(ids)
    has_hand = {sid: (sid in data['hands']) for sid in ids}

    obj_verts = data['obj_verts']
    obj_faces = data['obj_faces']
    bbox_corners = data['obj_bbox_corners']
    bbox_h = np.column_stack([bbox_corners, np.ones(8)])

    # ---- viser server ----
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"\n  ==>  http://localhost:{args.port}  <==\n")

    # ---- GUI ----
    with server.gui.add_folder("Scene"):
        gui_show_obj = server.gui.add_checkbox("Object mesh", initial_value=True)
        gui_show_bbox = server.gui.add_checkbox("Object bbox", initial_value=True)
        gui_show_hands = server.gui.add_checkbox("Hands", initial_value=True)
        gui_show_wrist = server.gui.add_checkbox("Wrist markers", initial_value=True)
        gui_obj_alpha = server.gui.add_slider(
            "Object alpha", min=0.1, max=1.0, step=0.05, initial_value=0.55)
        gui_hand_alpha = server.gui.add_slider(
            "Hand alpha", min=0.1, max=1.0, step=0.05, initial_value=0.75)

    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider(
            "Frame", min=0, max=N - 1, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Play", initial_value=True)
        gui_speed = server.gui.add_slider(
            "Speed", min=1, max=60, step=1, initial_value=args.fps)

    with server.gui.add_folder("Info"):
        gui_info = server.gui.add_text("Frame", initial_value="0 / 0", disabled=True)
        gui_stats = server.gui.add_text("Hands", initial_value="", disabled=True)

    # Reference frame at camera origin
    server.scene.add_frame("origin", show_axes=True, axes_length=0.15, axes_radius=0.005)

    # Grid at ~1m depth (typical table distance)
    server.scene.add_grid("floor", width=3.0, height=3.0,
                          position=to_viser(np.array([0, 0, 1.0])),
                          cell_size=0.2)

    # -----------------------------------------------------------------------
    # Scene update
    # -----------------------------------------------------------------------

    def update_scene():
        i = int(gui_frame.value)
        sid = ids[i]
        sid_has_hand = has_hand.get(sid, False)

        # ---- object mesh (always available for all frames) ----
        if gui_show_obj.value and sid in data['poses']:
            pose = data['poses'][sid]
            v_h = np.column_stack([obj_verts, np.ones(len(obj_verts))])
            v_posed = (pose @ v_h.T).T[:, :3]
            server.scene.add_mesh_simple(
                "object", vertices=to_viser(v_posed), faces=obj_faces,
                color=(80, 200, 100), opacity=float(gui_obj_alpha.value),
                wireframe=False, flat_shading=False,
            )
        else:
            server.scene.add_mesh_simple(
                "object", vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
                color=(80, 200, 100), opacity=0.0,
            )

        # ---- bounding box ----
        if gui_show_bbox.value and sid in data['poses']:
            pose = data['poses'][sid]
            bbox_posed = (pose @ bbox_h.T).T[:, :3]
            bv = to_viser(bbox_posed)
            segs = np.array([[bv[a], bv[b]] for a, b in _BBOX_EDGES], dtype=np.float32)
            server.scene.add_line_segments(
                "bbox", points=segs,
                colors=(0, 255, 0), line_width=2.0,
            )
        else:
            server.scene.add_line_segments(
                "bbox", points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=(0, 255, 0), line_width=0.0,
            )

        # ---- hands (may be missing on most frames) ----
        hand_shown = 0
        for h in range(2):
            server.scene.add_mesh_simple(
                f"hand_{h}", vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32), color=(0, 0, 0), opacity=0.0,
            )
            server.scene.add_point_cloud(
                f"wrist_{h}", points=np.zeros((0, 3), dtype=np.float32),
                colors=(0, 0, 0), point_size=0.0,
            )

        if gui_show_hands.value and sid_has_hand:
            hd = data['hands'][sid]
            if hd.get('verts_mano') is not None and hd.get('n_hands') != 0:
                vm = np.asarray(hd['verts_mano'], dtype=np.float32)   # full 3D shape
                jt = np.asarray(hd['joints'], dtype=np.float32)       # [N, 21, 3]
                ir = np.asarray(hd['is_right'], dtype=np.uint8).reshape(-1)
                wr = np.asarray(hd['wrist_3d'], dtype=np.float32) if hd.get('wrist_3d') is not None else np.zeros((len(vm), 3))
                for h_idx in range(min(len(vm), 2)):
                    # verts_mano has full 3D structure but in WiLoR space.
                    # Anchor at the wrist joint (joints[0]) and place at
                    # the metric wrist_3d position for correct 3D location.
                    wrist_mano = jt[h_idx, 0]  # (3,) wrist in WiLoR space
                    v_display = vm[h_idx] - wrist_mano + wr[h_idx]
                    clr = (0, 200, 220) if ir[h_idx] else (255, 140, 0)
                    server.scene.add_mesh_simple(
                        f"hand_{h_idx}", vertices=to_viser(v_display),
                        faces=mano_faces, color=clr,
                        opacity=float(gui_hand_alpha.value),
                        wireframe=False, flat_shading=False,
                    )
                    if gui_show_wrist.value and np.any(wr[h_idx] != 0):
                        server.scene.add_point_cloud(
                            f"wrist_{h_idx}", points=to_viser(wr[h_idx].reshape(1, 3)),
                            colors=(255, 0, 0), point_size=0.012,
                        )
                    hand_shown += 1

        # ---- info panel ----
        hand_note = f"Hands: {hand_shown}" if sid_has_hand else "Hands: --"
        gui_info.value = f"{sid}  ({i + 1} / {N})"
        gui_stats.value = hand_note

    # -----------------------------------------------------------------------
    # Animation thread
    # -----------------------------------------------------------------------

    _lock = threading.Lock()
    _last_tick = time.time()

    def anim_loop():
        nonlocal _last_tick
        while True:
            time.sleep(0.025)
            if gui_playing.value:
                now = time.time()
                interval = 1.0 / max(float(gui_speed.value), 1)
                if now - _last_tick >= interval:
                    with _lock:
                        nxt = (int(gui_frame.value) + 1) % N
                        gui_frame.value = nxt
                    _last_tick = now

    threading.Thread(target=anim_loop, daemon=True).start()

    # ---- wire callbacks ----
    gui_frame.on_update(lambda _: update_scene())
    gui_show_obj.on_update(lambda _: update_scene())
    gui_show_bbox.on_update(lambda _: update_scene())
    gui_show_hands.on_update(lambda _: update_scene())
    gui_show_wrist.on_update(lambda _: update_scene())
    gui_obj_alpha.on_update(lambda _: update_scene())
    gui_hand_alpha.on_update(lambda _: update_scene())

    # initial draw
    update_scene()

    print(f"Ready — {N} frames loaded.  Use the browser to orbit / pan / zoom.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Done.")


if __name__ == '__main__':
    main()
