"""FoundationPose 6DoF tracking with FFS depth maps.

Must be run inside the FoundationPose Docker container.
"""

import argparse
import logging
import os
import sys
import numpy as np

from .._pathresolver import paths
from .data_reader import FPDataReader


def run(clip: str, camera: str = "left", *,
        shorter_side: int = 800,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        start_frame: int = 0,
        end_frame: int = -1,
        debug: int = 1,
        debug_dir: str | None = None,
        zfar: float = 2.0,
        mesh_scale: float | None = None,
        ) -> None:
    """Run FoundationPose tracking on a clip for one camera view.

    Args:
        clip:              clip name.
        camera:            ``'left'`` or ``'right'``.
        shorter_side:      resize shorter side to this (pixels) before tracking.
        est_refine_iter:   refinement iterations for the first-frame registration.
        track_refine_iter: refinement iterations per tracking step.
        start_frame:       first frame index.
        end_frame:         last frame index (exclusive; -1 = all).
        debug:             ``0`` = none, ``1`` = show window, ``2`` = save images.
        debug_dir:         output directory (default: ``foundationpose_v2/run[_right]``).
        zfar:              far-plane depth in metres.
        mesh_scale:        object mesh scale factor (auto-detected if None).
    """
    # Lazy import from FoundationPose submodule
    sys.path.insert(0, str(paths.foundationpose_dir))
    from estimater import (
        FoundationPose, ScorePredictor, PoseRefinePredictor,
        set_logging_format, set_seed,
        draw_posed_3d_box, draw_xyz_axis, nvdiffrast_render,
    )
    import trimesh
    import torch
    import nvdiffrast.torch as dr
    import imageio

    set_logging_format()
    set_seed(0)

    data_dir = str(paths.clip_dir(clip))
    mesh_file = os.path.join(data_dir, "mesh", "clean_mesh.obj")
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh not found: {mesh_file}")

    # ---- output dir ----
    if debug_dir is None:
        suffix = "_right" if camera == "right" else ""
        debug_dir = os.path.join(data_dir, "foundationpose_v2", f"run{suffix}")
    os.makedirs(os.path.join(debug_dir, "track_vis"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ob_in_cam"), exist_ok=True)

    # ---- mesh scale ----
    if mesh_scale is not None:
        scale = mesh_scale
    else:
        scale_path = os.path.join(data_dir, "foundationpose", "run",
                                   "scales", "unified_scale.txt")
        scale = float(open(scale_path).read().strip()) if os.path.exists(scale_path) else 1.0
    logging.info("Mesh scale: %s", scale)

    # ---- mesh ----
    mesh = trimesh.load(mesh_file)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    diag_before = np.linalg.norm(
        mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    mesh.vertices = mesh.vertices * scale
    diag_after = np.linalg.norm(
        mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    logging.info("Mesh: %d verts, diagonal %.3f m → %.3f m",
                 len(mesh.vertices), diag_before, diag_after)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ---- FoundationPose ----
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir=debug_dir, debug=debug, glctx=glctx,
    )
    logging.info("[%s] FoundationPose ready", camera)

    # ---- reader ----
    reader = FPDataReader(data_dir, camera=camera,
                          shorter_side=shorter_side, zfar=zfar)

    _end = end_frame if end_frame > 0 else len(reader)
    _end = min(_end, len(reader))
    logging.info("[%s] Frames %d → %d / %d",
                 camera, start_frame, _end - 1, len(reader))

    for i in range(start_frame, _end):
        logging.info("[%s] Frame %d (%s)", camera, i, reader.id_strs[i])

        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == start_frame:
            mask = reader.get_mask(i).astype(bool)
            if mask.sum() < 100:
                logging.warning("Frame %d: mask has only %d pixels", i, mask.sum())
            pose = est.register(K=reader.K, rgb=color, depth=depth,
                                ob_mask=mask, iteration=est_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                                 iteration=track_refine_iter)

        np.savetxt(os.path.join(debug_dir, "ob_in_cam",
                                 f"{reader.id_strs[i]}.txt"),
                   pose.reshape(4, 4))

        # ---- visualisation ----
        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)

            vis_rgb = draw_posed_3d_box(
                reader.K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
            vis_rgb = draw_xyz_axis(
                vis_rgb, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                thickness=3, transparency=0, is_input_rgb=True)

            if debug == 1:
                from cv2 import imshow, waitKey
                imshow(f"FP [{camera}]", vis_rgb[..., ::-1])
                if waitKey(1) == ord("q"):
                    break

            elif debug >= 2:
                # Mesh render (nvdiffrast requires H/W divisible by 8)
                H8 = (reader.H + 7) // 8 * 8
                W8 = (reader.W + 7) // 8 * 8
                K8 = reader.K.copy()
                K8[0] *= W8 / reader.W
                K8[1] *= H8 / reader.H

                ob_in_cam_t = torch.as_tensor(
                    pose, device="cuda", dtype=torch.float).reshape(1, 4, 4)
                rendered, _, _ = nvdiffrast_render(
                    K=K8, H=H8, W=W8,
                    ob_in_cams=ob_in_cam_t, glctx=est.glctx,
                    mesh_tensors=est.mesh_tensors, mesh=mesh,
                )
                rendered_np = (rendered[0].data.cpu().numpy() * 255).astype(np.uint8)
                rendered_np = rendered_np[:reader.H, :reader.W, :3].copy()

                vis_track = draw_posed_3d_box(
                    reader.K, img=rendered_np, ob_in_cam=center_pose, bbox=bbox)
                vis_track = draw_xyz_axis(
                    vis_track, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                    thickness=3, transparency=0, is_input_rgb=True)
                imageio.imwrite(
                    os.path.join(debug_dir, "track_vis",
                                 f"{reader.id_strs[i]}.png"),
                    vis_track,
                )

                vis_rgb_mesh = draw_posed_3d_box(
                    reader.K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
                vis_rgb_mesh = draw_xyz_axis(
                    vis_rgb_mesh, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                    thickness=3, transparency=0, is_input_rgb=True)
                os.makedirs(os.path.join(debug_dir, "video_frames"),
                            exist_ok=True)
                imageio.imwrite(
                    os.path.join(debug_dir, "video_frames",
                                 f"{reader.id_strs[i]}.png"),
                    vis_rgb_mesh,
                )

    logging.info("[%s] Done → %s/ob_in_cam/", camera, debug_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FoundationPose tracking with FFS depth")
    parser.add_argument("--clip", type=str, default="clip03")
    parser.add_argument("--camera", type=str, default="left",
                        choices=["left", "right"])
    parser.add_argument("--shorter_side", type=int, default=800)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=None)
    parser.add_argument("--zfar", type=float, default=2.0)
    parser.add_argument("--mesh_scale", type=float, default=None)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
