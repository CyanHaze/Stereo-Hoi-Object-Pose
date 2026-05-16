#!/usr/bin/env python
"""
Batch WiLoR hand mesh inference + metric 3D alignment.

Pipeline per frame:
  1. YOLO hand detection → bounding boxes
  2. WiLoR inference → MANO vertices (meters), joints, weak-persp camera
  3. Convert weak-persp → full-image camera translation (virtual units)
  4. Metric alignment: scale virtual→metric so that
     (a) the hand root depth matches FFS stereo depth, and
     (b) the MANO hand size (~0.18m) projects correctly through the real K.

  The WiLoR virtual camera uses focal length f_virt = 5000 * max(w,h) / 256.
  The real camera uses K.  We solve for anisotropic scale (S_x, S_y, S_z)
  that maps virtual → metric while preserving the 2D projection.

  5. Render the metric hand mesh (filled triangles) overlaid on RGB.

Output:
    data/<clip>/wilor/left/*.npz    — per-frame hand data + metric verts
    data/<clip>/wilor/video_frames/ — mesh overlay PNGs (--debug)

Usage:
    conda activate diffusion
    cd F:/Research/02_Projects/SRTP/Reproduction/WiLoR
    python ../scripts/run_wilor_hand.py --clip clip03 --camera left --end_frame 10 --debug
"""

import argparse, os, sys, glob, logging, time

WILOR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'WiLoR')
os.chdir(WILOR_DIR)
sys.path.insert(0, '.')

import torch, cv2, numpy as np

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from ultralytics import YOLO
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset

REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# ---------------------------------------------------------------------------
# cam_crop_to_full
# ---------------------------------------------------------------------------
def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


# ---------------------------------------------------------------------------
# Metric alignment: WiLoR virtual → real camera
# ---------------------------------------------------------------------------

def align_virtual_to_metric(cam_full, verts_mano, depth_map, K, img_w, img_h, f_virt):
    """Convert WiLoR virtual-camera coordinates to metric camera frame.

    WiLoR uses a virtual camera with focal length f_virt (≈37500 for 1920-wide).
    The MANO vertices are in meters, positioned at cam_full [tx,ty,tz] in the
    virtual camera frame.  We convert to real camera frame (K with fx≈1505).

    Two outputs:
      verts_virt  — for RENDERING: use virtual camera (K_virt with f_virt).
                    Guarantees correct 2D overlay (same projection as WiLoR).
      wrist_3d    — metric wrist position from FFS depth (for downstream use).
      verts_metric — metric 3D vertices in real camera frame.

    Returns: (verts_metric, wrist_3d, verts_virt, ok)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # --- Vertices in WiLoR virtual camera frame (for rendering) ---
    verts_virt = verts_mano + cam_full.astype(np.float32)

    # --- Wrist 2D from WiLoR projection ---
    wrist_px = np.array([
        f_virt * cam_full[0] / (cam_full[2] + 1e-9) + img_w / 2.,
        f_virt * cam_full[1] / (cam_full[2] + 1e-9) + img_h / 2.
    ])

    # --- Metric depth from FFS at hand region ---
    px_all = (f_virt * verts_virt[:, :2] / (verts_virt[:, 2:3] + 1e-9)
              + np.array([img_w / 2., img_h / 2.])).astype(int)
    u_min = max(0, px_all[:, 0].min() - 3)
    u_max = min(img_w - 1, px_all[:, 0].max() + 3)
    v_min = max(0, px_all[:, 1].min() - 3)
    v_max = min(img_h - 1, px_all[:, 1].max() + 3)

    H_d, W_d = depth_map.shape
    if u_max > u_min and v_max > v_min:
        region = depth_map[max(0, v_min):min(H_d, v_max + 1),
                           max(0, u_min):min(W_d, u_max + 1)]
        valid_d = region[(region > 0.1) & np.isfinite(region)]
    else:
        valid_d = np.array([])

    if len(valid_d) < 5:
        verts_metric = verts_virt.copy()
        return verts_metric, np.zeros(3, dtype=np.float32), verts_virt.copy(), False

    Z_hand = float(np.median(valid_d))

    # --- Metric wrist 3D ---
    wrist_3d = np.array([
        (wrist_px[0] - cx) * Z_hand / fx,
        (wrist_px[1] - cy) * Z_hand / fy,
        Z_hand
    ], dtype=np.float32)

    # --- Metric vertices: scale virtual → real camera frame ---
    # In virtual cam:  pixel = f_virt * X_v / Z_v + img_center
    # In real cam:     pixel = fx * X_r / Z_r + cx
    # For the same pixel: X_r = X_v * (Z_r / Z_v) * (f_virt / fx)
    tz_virt = float(cam_full[2])
    s_z = Z_hand / tz_virt
    s_xy = (f_virt / fx) * s_z

    verts_metric = verts_virt.copy()
    verts_metric[:, 0] = verts_virt[:, 0] * s_xy
    verts_metric[:, 1] = verts_virt[:, 1] * s_xy
    verts_metric[:, 2] = verts_virt[:, 2] * s_z
    # Shift origin so wrist lands at wrist_3d
    wrist_in_scaled = np.array([cam_full[0] * s_xy, cam_full[1] * s_xy,
                                 cam_full[2] * s_z], dtype=np.float32)
    verts_metric = verts_metric - wrist_in_scaled + wrist_3d

    return verts_metric, wrist_3d, verts_virt.copy(), True


# ---------------------------------------------------------------------------
# Mesh rendering
# ---------------------------------------------------------------------------

def render_mesh_overlay(img_bgr, verts_cam, faces, K, color=(200, 180, 220), alpha=0.5):
    """Filled-triangle mesh overlay with depth-aware alpha blending."""
    H, W = img_bgr.shape[:2]
    p = K @ verts_cam.T
    z = p[2]
    p = (p[:2] / (z + 1e-9)).T

    # Keep faces that are in front of camera and not too spread out (avoid background faces)
    z_ok = (z[faces] > 0.01).all(axis=1)
    in_img = ((p[faces] >= -50) & (p[faces] < np.array([W + 50, H + 50]))).all(axis=(1, 2))
    too_big = (np.abs(p[faces]).max(axis=1).max(axis=-1) < max(W, H) * 3)
    valid = z_ok & in_img & too_big

    if valid.sum() == 0:
        return img_bgr

    tri_pts = p[faces[valid]].astype(np.int32)

    overlay = img_bgr.copy()
    cv2.fillPoly(overlay, tri_pts, color, lineType=cv2.LINE_AA)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, tri_pts, 255, lineType=cv2.LINE_AA)
    mb = mask > 0
    img_bgr[mb] = (img_bgr[mb] * (1 - alpha) + overlay[mb] * alpha).astype(np.uint8)
    return img_bgr


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_models(device):
    logging.info("Loading WiLoR...")
    model, cfg = load_wilor('./pretrained_models/wilor_final.ckpt',
                            './pretrained_models/model_config.yaml')
    model = model.to(device).eval()
    logging.info("Loading detector...")
    detector = YOLO('./pretrained_models/detector.pt').to(device)
    logging.info("Ready.")
    return model, cfg, detector


def process_frame(model, cfg, detector, img_bgr, device, conf=0.3):
    detections = detector(img_bgr, conf=conf, verbose=False)[0]
    bboxes, is_right_list = [], []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())

    if len(bboxes) == 0:
        return None

    boxes = np.stack(bboxes)
    right = np.stack(is_right_list)
    dataset = ViTDetDataset(cfg, img_bgr, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,
                                             num_workers=0)

    img_h, img_w = img_bgr.shape[:2]
    # WiLoR native focal length at model resolution (256px) is 5000.
    # For full-image rendering, scale it: f_scaled = 5000 * max(w,h) / 256
    f_scaled = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(img_w, img_h)

    all_verts, all_joints, all_joints_2d = [], [], []
    all_cam_crop, all_cam_full, all_is_right = [], [], []

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        # CRITICAL: pass f_scaled (NOT the default 5000) to cam_crop_to_full.
        # The demo uses scaled_focal_length for both cam_crop_to_full and
        # the renderer. Using 5000 makes tz 7.5x too small → hand appears
        # 7.5x too large.
        cam_full = cam_crop_to_full(pred_cam,
                                     batch["box_center"].float(),
                                     batch["box_size"].float(),
                                     batch["img_size"].float(),
                                     focal_length=f_scaled)

        for n in range(batch['img'].shape[0]):
            v = out['pred_vertices'][n].cpu().numpy()
            j = out['pred_keypoints_3d'][n].cpu().numpy()
            is_r = batch['right'][n].cpu().numpy()
            v[:, 0] = (2 * is_r - 1) * v[:, 0]
            j[:, 0] = (2 * is_r - 1) * j[:, 0]

            cf = cam_full[n].cpu().numpy()
            j3d_virt = j + cf
            j2d_x = f_scaled * j3d_virt[:, 0] / (j3d_virt[:, 2] + 1e-9) + img_w / 2.
            j2d_y = f_scaled * j3d_virt[:, 1] / (j3d_virt[:, 2] + 1e-9) + img_h / 2.
            j2d = np.stack([j2d_x, j2d_y], axis=-1)

            all_verts.append(v)
            all_joints.append(j)
            all_joints_2d.append(j2d)
            all_cam_crop.append(pred_cam[n].cpu().numpy())
            all_cam_full.append(cf)
            all_is_right.append(is_r)

    return {
        'verts_mano': np.stack(all_verts).astype(np.float32),
        'joints': np.stack(all_joints).astype(np.float32),
        'joints_2d': np.stack(all_joints_2d).astype(np.float32),
        'cam_crop': np.stack(all_cam_crop).astype(np.float32),
        'cam_full': np.stack(all_cam_full).astype(np.float32),
        'is_right': np.array(all_is_right).astype(np.uint8),
        'scaled_f': f_scaled,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='WiLoR hand mesh → metric 3D')
    parser.add_argument('--clip', type=str, default='clip03')
    parser.add_argument('--camera', type=str, default='left',
                        choices=['left', 'right', 'both'])
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=-1)
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    device = torch.device('cuda')
    model, cfg, detector = load_models(device)
    mano_faces = model.mano.faces  # (1538, 3)

    data_dir = os.path.join(REPO_DIR, 'data', args.clip)
    K = np.loadtxt(os.path.join(data_dir, 'ffs', 'cam_K.txt')).reshape(3, 3)

    cameras = ['left', 'right'] if args.camera == 'both' else [args.camera]

    for cam in cameras:
        rgb_dir = os.path.join(data_dir, 'rgb' if cam == 'left' else 'right')
        depth_dir = os.path.join(data_dir, 'ffs', 'depth')
        out_dir = os.path.join(data_dir, 'wilor', cam)
        vis_dir = os.path.join(data_dir, 'wilor', 'video_frames') if args.debug else None
        os.makedirs(out_dir, exist_ok=True)
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)

        color_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
        if not color_files:
            color_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        id_strs = [os.path.splitext(os.path.basename(f))[0] for f in color_files]
        total = len(id_strs)
        start, end = args.start_frame, (args.end_frame if args.end_frame > 0 else total)
        end = min(end, total)

        todo = sum(1 for i in range(start, end)
                   if args.overwrite or
                   not os.path.exists(os.path.join(out_dir, f'{id_strs[i]}.npz')))

        logging.info(f"[{cam}] {start}–{end - 1} / {total} | todo={todo}")
        if todo == 0:
            logging.info(f"[{cam}] All done.")
            continue

        t0 = time.time(); done = 0
        for i in range(start, end):
            sid = id_strs[i]
            out_path = os.path.join(out_dir, f'{sid}.npz')
            if os.path.exists(out_path) and not args.overwrite:
                continue

            img_bgr = cv2.imread(color_files[i])
            if img_bgr is None:
                continue
            result = process_frame(model, cfg, detector, img_bgr, device, conf=args.conf)

            if result is not None and len(result['verts_mano']) > 0:
                N = len(result['verts_mano'])
                img_h, img_w = img_bgr.shape[:2]
                f_virt = result['scaled_f']

                # Load FFS depth
                depth_m = None
                depth_path = os.path.join(depth_dir, f'{sid}.png')
                if os.path.exists(depth_path):
                    depth_m = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0

                verts_cam_list, wrist_list, verts_virt_list, ok_list = [], [], [], []
                for h in range(N):
                    if depth_m is not None:
                        vc, wr, vv, ok = align_virtual_to_metric(
                            result['cam_full'][h], result['verts_mano'][h],
                            depth_m, K, img_w, img_h, f_virt)
                    else:
                        vc = result['verts_mano'][h] + result['cam_full'][h]
                        vv = vc.copy()
                        wr = np.zeros(3, dtype=np.float32)
                        ok = False
                    verts_cam_list.append(vc.astype(np.float32))
                    wrist_list.append(wr.astype(np.float32))
                    verts_virt_list.append(vv.astype(np.float32))
                    ok_list.append(ok)

                result['verts_cam'] = np.stack(verts_cam_list)
                result['verts_virt'] = np.stack(verts_virt_list)
                result['wrist_3d'] = np.stack(wrist_list)
                result['depth_ok'] = np.array(ok_list)

                save_dict = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
                np.savez_compressed(out_path, **save_dict)

                # Mesh overlay: use WiLoR VIRTUAL camera for correct 2D projection
                if args.debug:
                    K_virt = np.array([
                        [f_virt, 0, img_w / 2.],
                        [0, f_virt, img_h / 2.],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    vis = img_bgr.copy()
                    for h in range(N):
                        clr = [(255, 128, 0), (0, 255, 128)][result['is_right'][h]]
                        vis = render_mesh_overlay(vis, verts_virt_list[h],
                                                   mano_faces, K_virt, color=clr, alpha=0.5)
                    cv2.imwrite(os.path.join(vis_dir, f'{sid}.png'), vis)
            else:
                np.savez_compressed(out_path, n_hands=0)

            done += 1
            if done % 50 == 0:
                e = time.time() - t0
                logging.info(f"[{cam}] {done}/{todo}  ({e:.0f}s, {done / e:.1f} fps)")

        e = time.time() - t0
        logging.info(f"[{cam}] Done: {done} in {e:.0f}s → {out_dir}/")
    logging.info("All done.")


if __name__ == '__main__':
    main()
