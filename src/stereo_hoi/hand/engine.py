"""Batch WiLoR hand mesh inference + metric 3D alignment.

Requires the ``diffusion`` conda environment and WiLoR pretrained models.
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import cv2

from .._pathresolver import paths
from .alignment import (
    cam_crop_to_full, align_virtual_to_metric,
    filter_hand_detections, render_mesh_overlay,
)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_models(device: str = "cuda"):
    """Load WiLoR + YOLO detector."""
    wilor_dir = str(paths.wilor_dir)

    # Patch torch.load for older checkpoints
    import torch
    _orig_load = torch.load

    def _patched_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    # WiLoR needs its own directory as CWD for relative imports
    _cwd = os.getcwd()
    os.chdir(wilor_dir)
    sys.path.insert(0, ".")

    try:
        logging.info("Loading WiLoR...")
        from wilor.models import load_wilor
        model, cfg = load_wilor(
            "./pretrained_models/wilor_final.ckpt",
            "./pretrained_models/model_config.yaml",
        )
        model = model.to(device).eval()

        logging.info("Loading detector...")
        from ultralytics import YOLO
        detector = YOLO("./pretrained_models/detector.pt").to(device)

        return model, cfg, detector
    finally:
        os.chdir(_cwd)


# ---------------------------------------------------------------------------
# Single-frame inference
# ---------------------------------------------------------------------------

def process_frame(model, cfg, detector, img_bgr: np.ndarray,
                  device: str = "cuda", conf: float = 0.3,
                  max_hands: int = 2,
                  top_margin_ratio: float = 0.12) -> dict | None:
    """Run WiLoR on a single frame.

    Returns a dict with keys: ``verts_mano``, ``joints``, ``joints_2d``,
    ``cam_crop``, ``cam_full``, ``is_right``, ``scaled_f``,
    or ``None`` if no hands were detected.
    """
    detections = detector(img_bgr, conf=conf, verbose=False)[0]
    bboxes, is_right_list = [], []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:5].tolist())

    img_h, img_w = img_bgr.shape[:2]

    bboxes, is_right_list, _ = filter_hand_detections(
        bboxes, is_right_list, img_h, img_w,
        max_hands=max_hands, top_margin_ratio=top_margin_ratio,
    )

    if len(bboxes) == 0:
        return None

    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    import torch
    import torch.utils.data

    boxes = np.array([b[:4] for b in bboxes], dtype=np.float32)
    right = np.array(is_right_list, dtype=np.float32)
    dataset = ViTDetDataset(cfg, img_bgr, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0)

    f_scaled = (cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE
                * max(img_w, img_h))

    all_verts, all_joints, all_joints_2d = [], [], []
    all_cam_crop, all_cam_full, all_is_right = [], [], []

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = (2 * batch["right"] - 1)
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        cam_full = cam_crop_to_full(
            pred_cam,
            batch["box_center"].float(),
            batch["box_size"].float(),
            batch["img_size"].float(),
            focal_length=f_scaled,
        )

        for n in range(batch["img"].shape[0]):
            v = out["pred_vertices"][n].cpu().numpy()
            j = out["pred_keypoints_3d"][n].cpu().numpy()
            is_r = batch["right"][n].cpu().numpy()
            v[:, 0] = (2 * is_r - 1) * v[:, 0]
            j[:, 0] = (2 * is_r - 1) * j[:, 0]

            cf = cam_full[n].cpu().numpy()
            j3d_virt = j + cf
            j2d_x = f_scaled * j3d_virt[:, 0] / (j3d_virt[:, 2] + 1e-9) + img_w / 2.0
            j2d_y = f_scaled * j3d_virt[:, 1] / (j3d_virt[:, 2] + 1e-9) + img_h / 2.0
            j2d = np.stack([j2d_x, j2d_y], axis=-1)

            all_verts.append(v)
            all_joints.append(j)
            all_joints_2d.append(j2d)
            all_cam_crop.append(pred_cam[n].cpu().numpy())
            all_cam_full.append(cf)
            all_is_right.append(is_r)

    return {
        "verts_mano": np.stack(all_verts).astype(np.float32),
        "joints": np.stack(all_joints).astype(np.float32),
        "joints_2d": np.stack(all_joints_2d).astype(np.float32),
        "cam_crop": np.stack(all_cam_crop).astype(np.float32),
        "cam_full": np.stack(all_cam_full).astype(np.float32),
        "is_right": np.array(all_is_right).astype(np.uint8),
        "scaled_f": f_scaled,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run(clip: str, camera: str = "left", *,
        start_frame: int = 0,
        end_frame: int = -1,
        conf: float = 0.3,
        max_hands: int = 2,
        hand_top_margin: float = 0.12,
        debug: bool = False,
        overwrite: bool = False,
        ) -> None:
    """Run WiLoR hand mesh inference on a clip.

    Args:
        clip:             clip name.
        camera:           ``'left'`` or ``'right'``.
        start_frame:      first frame index.
        end_frame:        last frame index (exclusive; -1 = all).
        conf:             YOLO detection confidence threshold.
        max_hands:        max hands per frame.
        hand_top_margin:  positional filter margin (0 = off).
        debug:            save mesh-overlay PNGs.
        overwrite:        re-process frames with existing ``.npz`` files.
    """
    import glob

    device = "cuda"
    model, cfg, detector = load_models(device)
    mano_faces = model.mano.faces

    data_dir = str(paths.clip_dir(clip))
    K = np.loadtxt(os.path.join(data_dir, "ffs", "cam_K.txt")).reshape(3, 3)

    rgb_dir = os.path.join(data_dir, "rgb" if camera == "left" else "right")
    depth_dir = os.path.join(data_dir, "ffs", "depth")
    out_dir = os.path.join(data_dir, "wilor", camera)
    vis_dir = os.path.join(data_dir, "wilor", "video_frames") if debug else None
    os.makedirs(out_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    color_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    if not color_files:
        color_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in color_files]
    total = len(id_strs)
    _end = total if end_frame <= 0 else min(end_frame, total)

    todo = sum(
        1 for i in range(start_frame, _end)
        if overwrite
        or not os.path.exists(os.path.join(out_dir, f"{id_strs[i]}.npz"))
    )

    logging.info("[%s] %d–%d / %d | todo=%d", camera, start_frame, _end - 1,
                 total, todo)
    if todo == 0:
        logging.info("[%s] All done.", camera)
        return

    t0 = time.time()
    done = 0
    for i in range(start_frame, _end):
        sid = id_strs[i]
        out_path = os.path.join(out_dir, f"{sid}.npz")
        if os.path.exists(out_path) and not overwrite:
            continue

        img_bgr = cv2.imread(color_files[i])
        if img_bgr is None:
            continue

        result = process_frame(
            model, cfg, detector, img_bgr, device,
            conf=conf, max_hands=max_hands, top_margin_ratio=hand_top_margin,
        )

        if result is not None and len(result["verts_mano"]) > 0:
            N = len(result["verts_mano"])
            img_h, img_w = img_bgr.shape[:2]
            f_virt = result["scaled_f"]

            depth_m = None
            depth_path = os.path.join(depth_dir, f"{sid}.png")
            if os.path.exists(depth_path):
                depth_m = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0

            verts_cam_list, wrist_list, verts_virt_list, ok_list = \
                [], [], [], []
            for h in range(N):
                if depth_m is not None:
                    vc, wr, vv, ok = align_virtual_to_metric(
                        result["cam_full"][h], result["verts_mano"][h],
                        result["joints"][h],
                        depth_m, K, img_w, img_h, f_virt,
                    )
                else:
                    vc = result["verts_mano"][h].copy()
                    vv = vc + result["cam_full"][h]
                    wr = np.zeros(3, dtype=np.float32)
                    ok = False
                verts_cam_list.append(vc.astype(np.float32))
                wrist_list.append(wr.astype(np.float32))
                verts_virt_list.append(vv.astype(np.float32))
                ok_list.append(ok)

            result["verts_cam"] = np.stack(verts_cam_list)
            result["verts_virt"] = np.stack(verts_virt_list)
            result["wrist_3d"] = np.stack(wrist_list)
            result["depth_ok"] = np.array(ok_list)

            save_dict = {k: v for k, v in result.items()
                         if isinstance(v, np.ndarray)}
            np.savez_compressed(out_path, **save_dict)

            if debug:
                K_virt = np.array([
                    [f_virt, 0, img_w / 2.0],
                    [0, f_virt, img_h / 2.0],
                    [0, 0, 1],
                ], dtype=np.float32)
                vis = img_bgr.copy()
                for h in range(N):
                    clr = [(255, 128, 0), (0, 255, 128)][result["is_right"][h]]
                    vis = render_mesh_overlay(
                        vis, verts_virt_list[h], mano_faces,
                        K_virt, color=clr, alpha=0.5,
                    )
                cv2.imwrite(os.path.join(vis_dir, f"{sid}.png"), vis)
        else:
            np.savez_compressed(out_path, n_hands=0)

        done += 1
        if done % 50 == 0:
            e = time.time() - t0
            logging.info("[%s] %d/%d  (%.0f s, %.1f fps)",
                         camera, done, todo, e, done / e)

    e = time.time() - t0
    logging.info("[%s] Done: %d in %.0f s → %s/", camera, done, e, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WiLoR hand mesh → metric 3D")
    parser.add_argument("--clip", type=str, default="clip03")
    parser.add_argument("--camera", type=str, default="left")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--max_hands", type=int, default=2)
    parser.add_argument("--hand_top_margin", type=float, default=0.12)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(args.clip, args.camera,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        conf=args.conf,
        max_hands=args.max_hands,
        hand_top_margin=args.hand_top_margin,
        debug=args.debug,
        overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
