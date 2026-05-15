"""
FoundationPose 6DoF tracking with FFS depth maps.

Data flow:
  FFS batch (run_ffs_batch.py)
    → data/{clip}/ffs/depth/*.png  (uint16 mm, left-camera)
    → data/{clip}/ffs/cam_K.txt    (3x3 intrinsics)
    → data/{clip}/calib.json       (stereo baseline, used for right-camera warp)
    → data/{clip}/rgb/*.jpg        (left RGB)
    → data/{clip}/right/*.jpg      (right RGB)
    → data/{clip}/mask/object/*.png (object mask, left-camera)
    → data/{clip}/mesh/clean_mesh.obj

  FoundationPose (this script)
    → data/{clip}/foundationpose_v2/run/ob_in_cam/*.txt        (left poses)
    → data/{clip}/foundationpose_v2/run_right/ob_in_cam/*.txt  (right poses)

Usage (inside FoundationPose Docker container):
    cd /mnt/f/Research/02_Projects/SRTP/Reproduction

    # Left camera
    python scripts/run_fp_track.py --clip clip03 --camera left --debug 1 --end_frame 5

    # Right camera (depth auto-warped from left via baseline)
    python scripts/run_fp_track.py --clip clip03 --camera right --debug 1 --end_frame 5

    # Full sequence with visualization saved
    python scripts/run_fp_track.py --clip clip03 --camera left --debug 2
    python scripts/run_fp_track.py --clip clip03 --camera right --debug 2
"""

import argparse
import os, sys, json, glob

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(code_dir)
sys.path.insert(0, os.path.join(repo_dir, 'FoundationPose'))

from estimater import *
import cv2


# ---------------------------------------------------------------------------
# Depth warping: left → right camera (rectified stereo, no disparity needed)
# ---------------------------------------------------------------------------

def warp_left_depth_to_right(depth_left_m, K, baseline_m):
    """Forward-warp left-camera depth to right-camera view.

    Uses stereo geometry only: for a left pixel (u,v) at depth Z,
    the corresponding right-image coordinate is (u - fx*baseline/Z, v).

    Args:
        depth_left_m: (H, W) float32, depth in meters (left camera)
        K: (3, 3) camera intrinsics
        baseline_m: stereo baseline in meters

    Returns:
        depth_right_m: (H, W) float32, depth in meters (right camera)
    """
    H, W = depth_left_m.shape
    fx = K[0, 0]

    depth_right = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    valid = (depth_left_m > 0.001) & np.isfinite(depth_left_m)

    if valid.sum() == 0:
        return depth_right

    # Displacement from left to right pixel (positive = rightward in left image,
    # meaning the same point appears further LEFT in the right image)
    flow_x = fx * baseline_m / depth_left_m[valid]  # > 0
    xx_r = np.round(xx[valid] - flow_x).astype(np.int32)
    yy_r = yy[valid]

    in_bounds = (xx_r >= 0) & (xx_r < W)
    xx_r = xx_r[in_bounds]; yy_r = yy_r[in_bounds]
    d_vals = depth_left_m[valid][in_bounds]

    np.add.at(depth_right, (yy_r, xx_r), d_vals)
    np.add.at(weight, (yy_r, xx_r), 1.0)

    mask = weight > 0
    depth_right[mask] /= weight[mask]

    # Fill holes via nearest-neighbor inpainting
    missing = ~mask
    if missing.any():
        from scipy.ndimage import distance_transform_edt
        if mask.any():
            dist, idx = distance_transform_edt(missing, return_indices=True)
            depth_right = depth_right[tuple(idx)]

    return depth_right


def warp_mask_to_right(mask_left, K, baseline_m, depth_left_m):
    """Forward-warp binary mask from left to right camera view."""
    H, W = mask_left.shape
    fx = K[0, 0]

    mask_right = np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    valid = (mask_left > 0) & (depth_left_m > 0.001)

    if valid.sum() == 0:
        return mask_right

    flow_x = fx * baseline_m / depth_left_m[valid]
    xx_r = np.round(xx[valid] - flow_x).astype(np.int32)
    yy_r = yy[valid]
    in_bounds = (xx_r >= 0) & (xx_r < W)
    mask_right[yy_r[in_bounds], xx_r[in_bounds]] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_right = cv2.dilate(mask_right, kernel, iterations=1)
    return mask_right


# ---------------------------------------------------------------------------
# Data reader
# ---------------------------------------------------------------------------

class FPDataReader:
    """Reads RGB, FFS depth, mask, K for FoundationPose tracking.

    Supports left/right cameras. Right-camera depth & mask are warped
    from left-camera FFS depth using stereo baseline (no disparity needed).
    """

    def __init__(self, data_dir, camera='left', shorter_side=None, zfar=np.inf):
        self.data_dir = data_dir
        self.camera = camera
        self.zfar = zfar

        # RGB
        rgb_dir = 'rgb' if camera == 'left' else 'right'
        pattern = os.path.join(data_dir, rgb_dir, '*.jpg')
        self.color_files = sorted(glob.glob(pattern))
        if not self.color_files:
            pattern = os.path.join(data_dir, rgb_dir, '*.png')
            self.color_files = sorted(glob.glob(pattern))
        if not self.color_files:
            raise FileNotFoundError(f"No images in {data_dir}/{rgb_dir}/")

        self.id_strs = [os.path.splitext(os.path.basename(f))[0] for f in self.color_files]

        # Intrinsics
        K_path = os.path.join(data_dir, 'ffs', 'cam_K.txt')
        self.K = np.loadtxt(K_path).reshape(3, 3)

        # Baseline (for right-camera warp)
        calib_path = os.path.join(data_dir, 'calib.json')
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                self.baseline_m = float(json.load(f)['baseline_m'])
        else:
            self.baseline_m = 0.0

        # Resolution
        H, W = cv2.imread(self.color_files[0]).shape[:2]
        self.H_orig, self.W_orig = H, W
        if shorter_side is not None:
            downscale = shorter_side / min(H, W)
            self.H = int(H * downscale)
            self.W = int(W * downscale)
            self.K = self.K.copy()
            self.K[:2] *= downscale
        else:
            self.H, self.W = H, W

        logging.info(f"[{camera}] {self.W}x{self.H}, {len(self)} frames, "
                     f"baseline={self.baseline_m:.4f}m")

    def __len__(self):
        return len(self.color_files)

    def get_color(self, i):
        color = imageio.imread(self.color_files[i])
        if len(color.shape) == 2:
            color = np.tile(color[..., None], (1, 1, 3))
        color = color[..., :3]
        return cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def get_depth(self, i):
        # Read left FFS depth (uint16 mm → meters)
        depth_path = os.path.join(self.data_dir, 'ffs', 'depth', f'{self.id_strs[i]}.png')
        depth_mm = cv2.imread(depth_path, -1).astype(np.float32)
        depth_m = depth_mm / 1000.0
        depth_m = cv2.resize(depth_m, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        if self.camera == 'right' and self.baseline_m > 0:
            depth_m = warp_left_depth_to_right(depth_m, self.K, self.baseline_m)

        depth_m[(depth_m < 0.001) | (depth_m >= self.zfar)] = 0
        return depth_m

    def get_mask(self, i):
        mask_path = os.path.join(self.data_dir, 'mask', 'object', f'{self.id_strs[i]}.png')
        if not os.path.exists(mask_path):
            return np.zeros((self.H, self.W), dtype=np.uint8)

        mask = cv2.imread(mask_path, -1)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        if self.camera == 'right' and self.baseline_m > 0:
            # Need left depth for the warp; read at original resolution
            depth_path = os.path.join(self.data_dir, 'ffs', 'depth', f'{self.id_strs[i]}.png')
            depth_left_mm = cv2.imread(depth_path, -1).astype(np.float32)
            depth_left = depth_left_mm / 1000.0
            depth_left = cv2.resize(depth_left, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            mask = warp_mask_to_right(mask, self.K, self.baseline_m, depth_left)

        return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FoundationPose tracking with FFS depth')
    parser.add_argument('--clip', type=str, default='clip03')
    parser.add_argument('--camera', type=str, default='left', choices=['left', 'right'])
    parser.add_argument('--shorter_side', type=int, default=800)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=-1)
    parser.add_argument('--debug', type=int, default=1,
                        help='0=none, 1=show window, 2=save track_vis, 3=save debug meshes')
    parser.add_argument('--debug_dir', type=str, default=None)
    parser.add_argument('--zfar', type=float, default=2.0)
    parser.add_argument('--mesh_scale', type=float, default=None)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    data_dir = os.path.join(repo_dir, 'data', args.clip)
    mesh_file = os.path.join(data_dir, 'mesh', 'clean_mesh.obj')
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh not found: {mesh_file}")

    # Output directory
    if args.debug_dir is None:
        suffix = '_right' if args.camera == 'right' else ''
        args.debug_dir = os.path.join(data_dir, 'foundationpose_v2', f'run{suffix}')
    debug_dir = args.debug_dir
    os.makedirs(os.path.join(debug_dir, 'track_vis'), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'ob_in_cam'), exist_ok=True)

    # Mesh scale
    if args.mesh_scale is not None:
        mesh_scale = args.mesh_scale
    else:
        scale_file = os.path.join(data_dir, 'foundationpose', 'run', 'scales', 'unified_scale.txt')
        mesh_scale = float(open(scale_file).read().strip()) if os.path.exists(scale_file) else 1.0
    logging.info(f"Mesh scale: {mesh_scale}")

    # Load & scale mesh
    mesh = trimesh.load(mesh_file)
    diag_before = np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    mesh.vertices = mesh.vertices * mesh_scale
    diag_after = np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    logging.info(f"Mesh: {len(mesh.vertices)} verts, diagonal {diag_before:.3f}m → {diag_after:.3f}m")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # Init FoundationPose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir=debug_dir, debug=args.debug, glctx=glctx,
    )
    logging.info(f"[{args.camera}] FoundationPose ready")

    # Reader
    reader = FPDataReader(data_dir, camera=args.camera, shorter_side=args.shorter_side,
                          zfar=args.zfar)

    end_frame = args.end_frame if args.end_frame > 0 else len(reader)
    end_frame = min(end_frame, len(reader))
    logging.info(f"[{args.camera}] Frames {args.start_frame} → {end_frame - 1} / {len(reader)}")

    for i in range(args.start_frame, end_frame):
        logging.info(f"[{args.camera}] Frame {i} ({reader.id_strs[i]})")

        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == args.start_frame:
            mask = reader.get_mask(i).astype(bool)
            if mask.sum() < 100:
                logging.warning(f"Frame {i}: mask has only {mask.sum()} pixels")
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                                iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                                 iteration=args.track_refine_iter)

        # Save
        np.savetxt(os.path.join(debug_dir, 'ob_in_cam', f'{reader.id_strs[i]}.txt'),
                   pose.reshape(4, 4))

        # Visualization
        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)
            if args.debug == 1:
                cv2.imshow(f'FP [{args.camera}]', vis[..., ::-1])
                if cv2.waitKey(1) == ord('q'):
                    break
            elif args.debug >= 2:
                imageio.imwrite(os.path.join(debug_dir, 'track_vis', f'{reader.id_strs[i]}.png'), vis)

    logging.info(f"[{args.camera}] Done → {debug_dir}/ob_in_cam/")


if __name__ == '__main__':
    main()
