"""Unified CLI for the stereo HOI pipeline.

Usage::

    stereo-hoi depth   <clip> [opts]
    stereo-hoi track   <clip> <camera> [opts]
    stereo-hoi fuse    <clip> [opts]
    stereo-hoi hand    <clip> [opts]
    stereo-hoi render  <clip> [opts]
    stereo-hoi viewer  <clip> [opts]
    stereo-hoi export  <clip> [opts]
    stereo-hoi video   <clip> [opts]
"""

import argparse
import sys


def cmd_depth(args):
    from .depth.engine import run
    run(args.clip,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        model_dir=args.model_dir,
        scale=args.scale,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        zfar=args.zfar,
        save_intermediate=args.save_intermediate,
        )


def cmd_track(args):
    from .tracking.engine import run
    run(args.clip, args.camera,
        shorter_side=args.shorter_side,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        debug=args.debug,
        debug_dir=args.debug_dir,
        zfar=args.zfar,
        mesh_scale=args.mesh_scale,
        )


def cmd_fuse(args):
    from .fusion.core import build_left_from_right, load_poses, save_poses, \
        fuse_average, fuse_left_main
    from .fusion.outlier import detect_outliers
    from .fusion.smooth import smooth_poses
    import logging, glob, json, os, numpy as np
    from ._pathresolver import paths

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")

    data_dir = str(paths.clip_dir(args.clip))
    left_dir = os.path.join(data_dir, "foundationpose_v2", "run", "ob_in_cam")
    right_dir = os.path.join(data_dir, "foundationpose_v2", "run_right",
                              "ob_in_cam")
    out_dir = os.path.join(data_dir, "foundationpose_v2", "fused", "ob_in_cam")

    with open(os.path.join(data_dir, "calib.json")) as f:
        baseline_m = float(json.load(f)["baseline_m"])
    logging.info("Baseline: %.2f mm", baseline_m * 1000)

    T_left_from_right = build_left_from_right(baseline_m)

    left_files = sorted(glob.glob(os.path.join(left_dir, "*.txt")))
    if not left_files:
        raise FileNotFoundError(f"No pose files in {left_dir}")
    id_strs = [os.path.splitext(os.path.basename(f))[0] for f in left_files]

    _end = args.end_frame if args.end_frame > 0 else len(id_strs)
    id_strs = id_strs[args.start_frame:_end]
    logging.info("Processing frames [%d, %d) — %d frames",
                 args.start_frame, _end, len(id_strs))

    poses_left, miss_left = load_poses(left_dir, id_strs)
    poses_right_raw, miss_right = load_poses(right_dir, id_strs)

    poses_right = poses_right_raw.copy()
    for i in range(len(id_strs)):
        if not miss_right[i]:
            poses_right[i] = T_left_from_right @ poses_right_raw[i]

    valid = ~miss_left & ~miss_right

    if not args.no_outlier:
        logging.info("Outlier rejection: trans>%.0fmm | rot>%.0fdeg",
                     args.outlier_trans * 1000, args.outlier_rot)
        valid = detect_outliers(poses_left, poses_right, valid,
                                trans_thresh=args.outlier_trans,
                                rot_thresh_deg=args.outlier_rot)

    if args.method == "average":
        fused = fuse_average(poses_left, poses_right, valid)
    elif args.method == "left_main":
        fused = fuse_left_main(poses_left, poses_right, valid)
    elif args.method == "left_only":
        fused = poses_left.copy()
    elif args.method == "right_only":
        fused = poses_right.copy()
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Stats
    diffs_LR = []
    for i in range(len(valid)):
        if valid[i]:
            diffs_LR.append(np.linalg.norm(
                poses_left[i, :3, 3] - poses_right[i, :3, 3]))
    diffs_LR = np.array(diffs_LR) if diffs_LR else np.zeros(0)
    logging.info("Frames: %d total, %d fused, %d missing on one side",
                 len(valid), valid.sum(), (~valid).sum())
    logging.info("Left <-> Right translation discrepancy (mm): "
                 "mean=%.1f  max=%.1f",
                 float(diffs_LR.mean()) * 1000 if len(diffs_LR) else 0,
                 float(diffs_LR.max()) * 1000 if len(diffs_LR) else 0)

    if args.smooth > 0:
        logging.info("Temporal smoothing: window=%d, method=%s",
                     args.smooth, args.smooth_method)
        fused = smooth_poses(fused, window=args.smooth,
                             method=args.smooth_method)

    save_poses(out_dir, id_strs, fused)
    logging.info("Saved %d fused poses → %s", len(id_strs), out_dir)

    if args.vis:
        from .vis.compare_video import run as run_vis
        # Generate visualisation via the same FoundationPose rendering
        logging.warning(
            "--vis requires FoundationPose Docker. "
            "Use 'stereo-hoi video --views fused' separately.")


def cmd_hand(args):
    from .hand.engine import run
    run(args.clip, args.camera,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        conf=args.conf,
        max_hands=args.max_hands,
        hand_top_margin=args.hand_top_margin,
        debug=args.debug,
        overwrite=args.overwrite,
        )


def cmd_render(args):
    from .vis.hoi_render import run
    run(args.clip, args.camera,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        overwrite=args.overwrite,
        )


def cmd_viewer(args):
    from .vis.viewer import run
    run(args.clip, host=args.host, port=args.port, fps=args.fps)


def cmd_export(args):
    from .vis.export_web import run
    run(args.clip, step=args.step, rgb=args.rgb, out=args.out)


def cmd_video(args):
    from .vis.compare_video import run
    run(args.clip, mode=args.mode, views=args.views, fps=args.fps,
        start_frame=args.start_frame, end_frame=args.end_frame)


# ---------------------------------------------------------------------------
# Subcommand builders
# ---------------------------------------------------------------------------

def _depth_subparser(sub):
    sub.add_argument("--clip", type=str, default="clip03")
    sub.add_argument("--start_frame", type=int, default=0)
    sub.add_argument("--end_frame", type=int, default=-1)
    sub.add_argument("--overwrite", action="store_true")
    sub.add_argument("--dry_run", action="store_true")
    sub.add_argument("--model_dir", type=str, default=None)
    sub.add_argument("--scale", type=float, default=1.0)
    sub.add_argument("--valid_iters", type=int, default=8)
    sub.add_argument("--max_disp", type=int, default=192)
    sub.add_argument("--zfar", type=float, default=100)
    sub.add_argument("--save_intermediate", action="store_true")


def _track_subparser(sub):
    sub.add_argument("--clip", type=str, default="clip03")
    sub.add_argument("--camera", type=str, default="left",
                     choices=["left", "right"])
    sub.add_argument("--shorter_side", type=int, default=800)
    sub.add_argument("--est_refine_iter", type=int, default=5)
    sub.add_argument("--track_refine_iter", type=int, default=2)
    sub.add_argument("--start_frame", type=int, default=0)
    sub.add_argument("--end_frame", type=int, default=-1)
    sub.add_argument("--debug", type=int, default=1)
    sub.add_argument("--debug_dir", type=str, default=None)
    sub.add_argument("--zfar", type=float, default=2.0)
    sub.add_argument("--mesh_scale", type=float, default=None)


def _fuse_subparser(sub):
    sub.add_argument("--clip", type=str, default="clip03")
    sub.add_argument("--method", type=str, default="average",
                     choices=["average", "left_main", "left_only", "right_only"])
    sub.add_argument("--outlier_trans", type=float, default=0.05)
    sub.add_argument("--outlier_rot", type=float, default=30.0)
    sub.add_argument("--no_outlier", action="store_true")
    sub.add_argument("--smooth", type=int, default=0)
    sub.add_argument("--smooth_method", type=str, default="gaussian",
                     choices=["gaussian", "moving_avg"])
    sub.add_argument("--vis", action="store_true")
    sub.add_argument("--start_frame", type=int, default=0)
    sub.add_argument("--end_frame", type=int, default=-1)


def _hand_subparser(sub):
    sub.add_argument("--clip", type=str, default="clip03")
    sub.add_argument("--camera", type=str, default="left")
    sub.add_argument("--start_frame", type=int, default=0)
    sub.add_argument("--end_frame", type=int, default=-1)
    sub.add_argument("--conf", type=float, default=0.3)
    sub.add_argument("--max_hands", type=int, default=2)
    sub.add_argument("--hand_top_margin", type=float, default=0.12)
    sub.add_argument("--debug", action="store_true")
    sub.add_argument("--overwrite", action="store_true")


def _render_subparser(sub):
    sub.add_argument("--clip", type=str, default="clip03")
    sub.add_argument("--camera", type=str, default="left")
    sub.add_argument("--start_frame", type=int, default=0)
    sub.add_argument("--end_frame", type=int, default=-1)
    sub.add_argument("--fps", type=int, default=0)
    sub.add_argument("--overwrite", action="store_true")


def _viewer_subparser(sub):
    sub.add_argument("--clip", type=str, default="clip03")
    sub.add_argument("--host", type=str, default="0.0.0.0")
    sub.add_argument("--port", type=int, default=8080)
    sub.add_argument("--fps", type=int, default=30)


def _export_subparser(sub):
    sub.add_argument("--clip", type=str, default="clip03")
    sub.add_argument("--step", type=int, default=1)
    sub.add_argument("--rgb", action="store_true")
    sub.add_argument("--out", type=str, default=None)


def _video_subparser(sub):
    sub.add_argument("--clip", type=str, default="clip03")
    sub.add_argument("--mode", type=str, default="both",
                     choices=["render", "compose", "both"])
    sub.add_argument("--views", type=str, default="left,fused")
    sub.add_argument("--fps", type=int, default=15)
    sub.add_argument("--start_frame", type=int, default=0)
    sub.add_argument("--end_frame", type=int, default=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMMANDS = {
    "depth":  (cmd_depth,  _depth_subparser,  "Run stereo depth estimation (FFS)"),
    "track":  (cmd_track,  _track_subparser,  "Run FoundationPose tracking on one camera"),
    "fuse":   (cmd_fuse,   _fuse_subparser,   "Fuse left+right poses (pure numpy, no GPU)"),
    "hand":   (cmd_hand,   _hand_subparser,   "Run WiLoR hand mesh inference"),
    "render": (cmd_render, _render_subparser, "Render 2D hand+object overlay frames"),
    "viewer": (cmd_viewer, _viewer_subparser, "Launch Viser 3D interactive viewer"),
    "export": (cmd_export, _export_subparser, "Export static web demo assets"),
    "video":  (cmd_video,  _video_subparser,  "Build comparison videos"),
}


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Stereo HOI object pose estimation pipeline")
    parser.add_argument("--version", action="version",
                        version="stereo-hoi 0.1.0")
    subp = parser.add_subparsers(dest="command", required=True)

    for name, (handler, builder, help_text) in COMMANDS.items():
        p = subp.add_parser(name, help=help_text, description=help_text)
        builder(p)
        p.set_defaults(_handler=handler)

    ns = parser.parse_args(args or sys.argv[1:])
    ns._handler(ns)


if __name__ == "__main__":
    main()
