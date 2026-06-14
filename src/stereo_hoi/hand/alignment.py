"""Pure helper functions for WiLoR virtual↔metric alignment and rendering."""

import logging
import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Camera transform
# ---------------------------------------------------------------------------

def cam_crop_to_full(cam_bbox: torch.Tensor,
                     box_center: torch.Tensor,
                     box_size: torch.Tensor,
                     img_size: torch.Tensor,
                     focal_length: float = 5000.0) -> torch.Tensor:
    """Convert WiLoR crop-camera params to full-image camera translation.

    Args:
        cam_bbox:    ``(N, 3)`` — [s, tx, ty] weak-perspective parameters.
        box_center:  ``(N, 2)`` — bounding box centre in pixel coords.
        box_size:    ``(N,)``   — bounding box size.
        img_size:    ``(N, 2)`` — [w, h] of the full image.
        focal_length: virtual camera focal length (scaled to image size).

    Returns:
        ``(N, 3)`` camera translation [tx, ty, tz] in virtual units.
    """
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


# ---------------------------------------------------------------------------
# Metric alignment
# ---------------------------------------------------------------------------

def align_virtual_to_metric(cam_full: np.ndarray,
                            verts_mano: np.ndarray,
                            joints: np.ndarray,
                            depth_map: np.ndarray,
                            K: np.ndarray,
                            img_w: int, img_h: int,
                            f_virt: float,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Place MANO hand at the correct metric 3D position using stereo depth.

    MANO vertices are already in real metres — no scaling, just
    translation to the metric wrist position obtained by back-projecting
    the WiLoR wrist pixel through the real camera at the FFS depth.

    Returns:
        ``(verts_metric, wrist_3d, verts_virt, ok)``
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    cam_full = cam_full.astype(np.float32)
    wrist_mano = joints[0].astype(np.float32)

    # Vertices in WiLoR virtual camera frame (for 2D overlay rendering)
    verts_virt = verts_mano + cam_full

    # Wrist pixel in virtual camera
    wrist_virt = wrist_mano + cam_full
    wrist_px = np.array([
        f_virt * wrist_virt[0] / (wrist_virt[2] + 1e-9) + img_w / 2.0,
        f_virt * wrist_virt[1] / (wrist_virt[2] + 1e-9) + img_h / 2.0,
    ])

    # Hand region in the depth map
    px_all = (
        f_virt * verts_virt[:, :2] / (verts_virt[:, 2:3] + 1e-9)
        + np.array([img_w / 2.0, img_h / 2.0])
    ).astype(int)
    u_min = max(0, int(px_all[:, 0].min()) - 3)
    u_max = min(img_w - 1, int(px_all[:, 0].max()) + 3)
    v_min = max(0, int(px_all[:, 1].min()) - 3)
    v_max = min(img_h - 1, int(px_all[:, 1].max()) + 3)

    H_d, W_d = depth_map.shape
    if u_max > u_min and v_max > v_min:
        region = depth_map[
            max(0, v_min):min(H_d, v_max + 1),
            max(0, u_min):min(W_d, u_max + 1),
        ]
        valid_d = region[(region > 0.1) & np.isfinite(region)]
    else:
        valid_d = np.array([])

    if len(valid_d) < 5:
        return verts_virt.copy(), np.zeros(3, dtype=np.float32), \
               verts_virt.copy(), False

    Z_hand = float(np.median(valid_d))

    wrist_3d = np.array([
        (wrist_px[0] - cx) * Z_hand / fx,
        (wrist_px[1] - cy) * Z_hand / fy,
        Z_hand,
    ], dtype=np.float32)

    verts_metric = verts_mano - wrist_mano + wrist_3d
    return verts_metric, wrist_3d, verts_virt.copy(), True


# ---------------------------------------------------------------------------
# Mesh overlay rendering
# ---------------------------------------------------------------------------

def render_mesh_overlay(img_bgr: np.ndarray,
                        verts_cam: np.ndarray,
                        faces: np.ndarray,
                        K: np.ndarray,
                        color: tuple[int, int, int] = (200, 180, 220),
                        alpha: float = 0.5) -> np.ndarray:
    """Filled-triangle mesh overlay with depth-aware alpha blending."""
    H, W = img_bgr.shape[:2]
    p = K @ verts_cam.T
    z = p[2]
    p = (p[:2] / (z + 1e-9)).T

    z_ok = (z[faces] > 0.01).all(axis=1)
    in_img = ((p[faces] >= -50) &
              (p[faces] < np.array([W + 50, H + 50]))).all(axis=(1, 2))
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
# Hand detection filter
# ---------------------------------------------------------------------------

def filter_hand_detections(bboxes: list,
                           is_right_list: list,
                           img_h: int, img_w: int,
                           max_hands: int = 2,
                           top_margin_ratio: float = 0.12,
                           ) -> tuple[list, list, int]:
    """Filter YOLO hand detections to keep only the main two hands.

    1. Drop detections whose centre is above ``top_margin_ratio``
       of the image height (likely faces / false positives).
    2. Group by left / right, keep the highest-confidence per hand type.
    3. If more than *max_hands* remain, trim by confidence.

    Returns ``(filtered_bboxes, filtered_is_right, n_filtered)``.
    """
    if len(bboxes) == 0:
        return bboxes, is_right_list, 0

    n_original = len(bboxes)

    # ---- positional filter ----
    y_thresh = img_h * top_margin_ratio
    pos_keep = []
    pos_drop = 0
    for idx, bbox in enumerate(bboxes):
        center_y = (bbox[1] + bbox[3]) / 2.0
        if center_y > y_thresh:
            pos_keep.append(idx)
        else:
            pos_drop += 1

    if len(pos_keep) == 0:
        pos_keep = list(range(n_original))
        pos_drop = 0

    if pos_drop > 0:
        logging.info("  [filter] position drop: %d det(s) above %.0f%% of image",
                     pos_drop, top_margin_ratio * 100)

    bboxes = [bboxes[i] for i in pos_keep]
    is_right_list = [is_right_list[i] for i in pos_keep]

    # ---- group by hand type ----
    def _conf(b):
        return float(b[4])

    entries = list(zip(bboxes, is_right_list))
    entries.sort(key=lambda e: _conf(e[0]), reverse=True)

    best_left = best_right = None
    for bbox, is_r in entries:
        if is_r == 0 and best_left is None:
            best_left = (bbox, is_r)
        elif is_r == 1 and best_right is None:
            best_right = (bbox, is_r)
        if best_left is not None and best_right is not None:
            break

    selected = []
    if best_left is not None:
        selected.append(best_left)
    if best_right is not None:
        selected.append(best_right)

    if len(selected) > max_hands:
        selected.sort(key=lambda e: _conf(e[0]), reverse=True)
        selected = selected[:max_hands]

    selected.sort(key=lambda e: e[1])  # left before right

    result_bboxes = [s[0] for s in selected]
    result_is_right = [s[1] for s in selected]
    n_filtered = n_original - len(result_bboxes)

    if n_filtered > 0:
        n_left = len([s for s in selected if s[1] == 0])
        n_right = len([s for s in selected if s[1] == 1])
        logging.info("  [filter] kept %d/%d hands (%d L, %d R)",
                     len(result_bboxes), n_original, n_left, n_right)

    return result_bboxes, result_is_right, n_filtered
