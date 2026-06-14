"""Interactive 3D HOI viewer via Viser.

Renders object mesh (at estimated pose) + hand meshes (WiLoR metric coords)
in a shared 3D scene with free orbit/pan/zoom camera, time slider,
play/pause, and per-element visibility controls.
"""

import argparse
import os
import sys
import threading
import time
import numpy as np

from .._pathresolver import paths
from ..hoi_data import load_data, load_mano_faces, to_viser, BBOX_EDGES


def run(clip: str, *, host: str = "0.0.0.0", port: int = 8080,
        fps: int = 30) -> None:
    """Launch the Viser interactive 3D viewer.

    Args:
        clip:  clip name.
        host:  bind address.
        port:  HTTP port.
        fps:   default playback speed.
    """
    import viser

    data_dir = str(paths.clip_dir(clip))

    print("Loading MANO faces...")
    mano_faces = load_mano_faces()

    print("Loading clip data...")
    data = load_data(data_dir)
    ids = data["pose_ids"]
    N = len(ids)
    has_hand = {sid: (sid in data["hands"]) for sid in ids}

    obj_verts = data["obj_verts"]
    obj_faces = data["obj_faces"]
    bbox_corners = data["obj_bbox_corners"]
    bbox_h = np.column_stack([bbox_corners, np.ones(8)])

    server = viser.ViserServer(host=host, port=port)
    print(f"\n  ==>  http://localhost:{port}  <==\n")

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
            "Speed", min=1, max=60, step=1, initial_value=fps)

    with server.gui.add_folder("Info"):
        gui_info = server.gui.add_text("Frame", initial_value="0 / 0", disabled=True)
        gui_stats = server.gui.add_text("Hands", initial_value="", disabled=True)

    # Reference frame at camera origin
    server.scene.add_frame("origin", show_axes=True, axes_length=0.15,
                           axes_radius=0.005)
    # Grid at ~1m depth
    server.scene.add_grid("floor", width=3.0, height=3.0,
                          position=to_viser(np.array([0, 0, 1.0])),
                          cell_size=0.2)

    def update_scene():
        i = int(gui_frame.value)
        sid = ids[i]
        sid_has_hand = has_hand.get(sid, False)

        # Object mesh
        if gui_show_obj.value and sid in data["poses"]:
            pose = data["poses"][sid]
            v_h = np.column_stack([obj_verts, np.ones(len(obj_verts))])
            v_posed = (pose @ v_h.T).T[:, :3]
            server.scene.add_mesh_simple(
                "object", vertices=to_viser(v_posed), faces=obj_faces,
                color=(80, 200, 100), opacity=float(gui_obj_alpha.value),
                wireframe=False, flat_shading=False,
            )
        else:
            server.scene.add_mesh_simple(
                "object",
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
                color=(80, 200, 100), opacity=0.0,
            )

        # Bounding box
        if gui_show_bbox.value and sid in data["poses"]:
            pose = data["poses"][sid]
            bbox_posed = (pose @ bbox_h.T).T[:, :3]
            bv = to_viser(bbox_posed)
            segs = np.array([[bv[a], bv[b]] for a, b in BBOX_EDGES],
                            dtype=np.float32)
            server.scene.add_line_segments(
                "bbox", points=segs, colors=(0, 255, 0), line_width=2.0,
            )
        else:
            server.scene.add_line_segments(
                "bbox",
                points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=(0, 255, 0), line_width=0.0,
            )

        # Hands
        for h in range(2):
            server.scene.add_mesh_simple(
                f"hand_{h}",
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
                color=(0, 0, 0), opacity=0.0,
            )
            server.scene.add_point_cloud(
                f"wrist_{h}",
                points=np.zeros((0, 3), dtype=np.float32),
                colors=(0, 0, 0), point_size=0.0,
            )

        hand_shown = 0
        if gui_show_hands.value and sid_has_hand:
            hd = data["hands"][sid]
            if hd.get("verts_mano") is not None and hd.get("n_hands") != 0:
                vm = np.asarray(hd["verts_mano"], dtype=np.float32)
                jt = np.asarray(hd["joints"], dtype=np.float32)
                ir = np.asarray(hd["is_right"], dtype=np.uint8).reshape(-1)
                wr = (np.asarray(hd["wrist_3d"], dtype=np.float32)
                      if hd.get("wrist_3d") is not None
                      else np.zeros((len(vm), 3)))
                for h_idx in range(min(len(vm), 2)):
                    wrist_mano = jt[h_idx, 0]
                    v_display = vm[h_idx] - wrist_mano + wr[h_idx]
                    clr = (0, 200, 220) if ir[h_idx] else (255, 140, 0)
                    server.scene.add_mesh_simple(
                        f"hand_{h_idx}",
                        vertices=to_viser(v_display),
                        faces=mano_faces, color=clr,
                        opacity=float(gui_hand_alpha.value),
                        wireframe=False, flat_shading=False,
                    )
                    if gui_show_wrist.value and np.any(wr[h_idx] != 0):
                        server.scene.add_point_cloud(
                            f"wrist_{h_idx}",
                            points=to_viser(wr[h_idx].reshape(1, 3)),
                            colors=(255, 0, 0), point_size=0.012,
                        )
                    hand_shown += 1

        gui_info.value = f"{sid}  ({i + 1} / {N})"
        gui_stats.value = f"Hands: {hand_shown}" if sid_has_hand else "Hands: --"

    # Animation thread
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

    gui_frame.on_update(lambda _: update_scene())
    gui_show_obj.on_update(lambda _: update_scene())
    gui_show_bbox.on_update(lambda _: update_scene())
    gui_show_hands.on_update(lambda _: update_scene())
    gui_show_wrist.on_update(lambda _: update_scene())
    gui_obj_alpha.on_update(lambda _: update_scene())
    gui_hand_alpha.on_update(lambda _: update_scene())

    update_scene()
    print(f"Ready — {N} frames loaded.  Use the browser to orbit / pan / zoom.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="HOI 3D viewer")
    parser.add_argument("--clip", type=str, default="clip03")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    run(args.clip, host=args.host, port=args.port, fps=args.fps)


if __name__ == "__main__":
    main()
