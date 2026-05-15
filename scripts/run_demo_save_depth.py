# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modified for headless batch processing:
#   - Reads K from 3-line format (actual cam_K.txt)
#   - Reads baseline from calib.json or --baseline_file
#   - Removed cv2.imshow / Open3D visualizer windows
#   - Added disp.npy saving
#   - Extracted load_ffs_model() / run_ffs_single() for batch reuse

import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(code_dir, '..', 'Fast-FoundationStereo'))
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, imageio, logging, yaml, json
import numpy as np
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
import cv2


def read_K_from_3line(txt_path):
    """Read 3x3 K matrix from a 3-line cam_K.txt file."""
    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    vals = []
    for line in lines[:3]:
        vals.extend([float(v) for v in line.split()])
    return np.array(vals, dtype=np.float32).reshape(3, 3)


def read_baseline_from_calib(calib_json_path):
    """Read baseline_m from calib.json."""
    with open(calib_json_path, 'r') as f:
        data = json.load(f)
    return float(data['baseline_m'])


def load_ffs_model(model_path):
    """Load FFS model once for reuse across frames.

    Args:
        model_path: path to model .pth file (e.g. weights/.../model_best_bp2_serialize.pth)

    Returns:
        model, cfg (OmegaConf dict with model args)
    """
    cfg_dir = os.path.dirname(model_path)
    with open(os.path.join(cfg_dir, 'cfg.yaml'), 'r') as ff:
        cfg = yaml.safe_load(ff)
    cfg = OmegaConf.create(cfg)

    logging.info(f"Loading FFS model from {model_path}")
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.cuda().eval()
    logging.info("FFS model loaded")
    return model, cfg


def run_ffs_single(model, cfg, left_file, right_file, K, baseline,
                   out_dir, scale=1.0, valid_iters=8, max_disp=192,
                   zfar=100, remove_invisible=True, get_pc=True,
                   denoise_cloud=False, denoise_nb_points=30,
                   denoise_radius=0.03, hiera=0):
    """Run FFS inference on a single stereo pair.

    Args:
        model: loaded FFS model (from load_ffs_model)
        cfg: model config (OmegaConf)
        left_file, right_file: paths to rectified left/right images
        K: 3x3 intrinsic matrix (numpy)
        baseline: stereo baseline in meters
        out_dir: directory to save outputs
        scale: image downscale factor
        valid_iters, max_disp, zfar: model parameters
        remove_invisible: filter occluded pixels
        get_pc: save point cloud
        denoise_cloud, denoise_nb_points, denoise_radius: denoising options
        hiera: use hierarchical mode

    Returns:
        dict with keys: depth_meter (np.array), disp (np.array),
            depth_meter_path, disp_path, cloud_path (str paths)
    """
    os.makedirs(out_dir, exist_ok=True)

    img0 = imageio.imread(left_file)
    img1 = imageio.imread(right_file)
    if len(img0.shape) == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]

    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    img1_ori = img1.copy()

    if get_pc:
        imageio.imwrite(os.path.join(out_dir, 'left.png'), img0)
        imageio.imwrite(os.path.join(out_dir, 'right.png'), img1)

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    logging.info("FFS forward pass ...")
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        if not hiera:
            disp = model.forward(img0_t, img1_t, iters=valid_iters,
                                 test_mode=True, optimize_build_volume='pytorch1')
        else:
            disp = model.run_hierachical(img0_t, img1_t, iters=valid_iters,
                                         test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

    # Save disparity
    disp_path = os.path.join(out_dir, 'disp.npy')
    np.save(disp_path, disp.astype(np.float32))

    # Save disparity visualization
    vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
    vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
    imageio.imwrite(os.path.join(out_dir, 'disp_vis.png'), vis)

    if remove_invisible:
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx - disp
        invalid = us_right < 0
        disp[invalid] = np.inf

    depth = None
    depth_path = None
    cloud_path = None

    if get_pc:
        K_scaled = K.copy()
        K_scaled[:2] *= scale
        depth = K_scaled[0, 0] * baseline / disp
        depth_path = os.path.join(out_dir, 'depth_meter.npy')
        np.save(depth_path, depth)

        xyz_map = depth2xyzmap(depth, K_scaled)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= zfar)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        cloud_path = os.path.join(out_dir, 'cloud.ply')
        o3d.io.write_point_cloud(cloud_path, pcd)

        if denoise_cloud:
            pcd = pcd.voxel_down_sample(voxel_size=0.001)
            cl, ind = pcd.remove_radius_outlier(nb_points=denoise_nb_points,
                                                radius=denoise_radius)
            inlier_cloud = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(os.path.join(out_dir, 'cloud_denoise.ply'), inlier_cloud)

    return {
        'depth_meter': depth,
        'disp': disp,
        'depth_meter_path': depth_path,
        'disp_path': disp_path,
        'cloud_path': cloud_path,
    }


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=f'{code_dir}/../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth', type=str)
    parser.add_argument('--left_file', required=True, type=str)
    parser.add_argument('--right_file', required=True, type=str)
    parser.add_argument('--intrinsic_file', required=True, type=str, help='cam_K.txt (3-line 3x3 matrix)')
    parser.add_argument('--baseline_file', default=None, type=str, help='calib.json for baseline')
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--remove_invisible', default=1, type=int)
    parser.add_argument('--denoise_cloud', default=0, type=int)
    parser.add_argument('--denoise_nb_points', type=int, default=30)
    parser.add_argument('--denoise_radius', type=float, default=0.03)
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--hiera', default=0, type=int)
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--zfar', type=float, default=100)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    K = read_K_from_3line(args.intrinsic_file)
    if args.baseline_file:
        baseline = read_baseline_from_calib(args.baseline_file)
    else:
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            baseline = float(lines[3])

    model, cfg = load_ffs_model(args.model_dir)

    result = run_ffs_single(
        model=model, cfg=cfg,
        left_file=args.left_file, right_file=args.right_file,
        K=K, baseline=baseline,
        out_dir=args.out_dir,
        scale=args.scale,
        valid_iters=args.valid_iters if hasattr(args, 'valid_iters') else 8,
        max_disp=args.max_disp if hasattr(args, 'max_disp') else 192,
        zfar=args.zfar,
        remove_invisible=bool(args.remove_invisible),
        get_pc=bool(args.get_pc),
        denoise_cloud=bool(args.denoise_cloud),
        denoise_nb_points=args.denoise_nb_points,
        denoise_radius=args.denoise_radius,
        hiera=bool(args.hiera),
    )

    logging.info("Done.")
