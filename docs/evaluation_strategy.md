# Evaluation and Benchmark Strategy

Last updated: 2026-06-14

## 1. Can Public Benchmarks Replace Ground Truth for `clip03`?

Yes for validating the research method, but no for measuring the absolute
accuracy of `clip03` itself.

The project should use two complementary tracks:

1. **Ground-truth benchmark track**: establish accuracy, robustness, and
   recovery claims on public datasets with object-pose annotations.
2. **In-the-wild stereo track**: use `clip03` for qualitative generalization,
   cross-view consistency, runtime, and documented failure cases.

No-GT consistency metrics must not be presented as substitutes for pose error.
They can compare variants on `clip03`, but they cannot prove that a pose is
metrically correct.

## 2. Benchmark Portfolio

| Dataset | Role | Strengths | Limitations for this project |
|---|---|---|---|
| HOT3D | Primary | Egocentric multi-view recordings, accurate hand/object/camera poses, object CAD models, visibility information, and BOP-compatible object-pose evaluation. | Data access agreement and substantial storage; official pose metrics are primarily frame-level and need sequence extensions. |
| H2O | Secondary multi-view | Four synchronized RGB-D views, two-hand/object interaction, 6D object pose, and action sequences. | Smaller object and scene diversity than HOT3D. |
| DexYCB | Secondary controlled multi-view | Eight calibrated RGB-D cameras, YCB objects, hand and object pose ground truth, broad subject/view coverage. | Controlled tabletop grasping; less representative of wearable stereo deployment. |
| HO3D | Single-view stress test | Established hand-object benchmark with severe occlusion and 6D object pose annotations. | Not a multi-view benchmark; useful for single-view comparison and transfer only. |
| BOP datasets | Pose-estimation compatibility | Standard MSSD, MSPD, VSD and Average Recall protocol with public toolkit. | Mostly frame-level pose estimation; not designed around long-term HOI tracking or recovery. |

ARCTIC and articulated-object datasets are useful future extensions, but they
change the task from rigid known-object tracking to articulated HOI and should
not be required for the first paper.

## 3. Evaluation Tasks

### Task A: Per-frame known-object 6D pose

- Input: synchronized calibrated views, object mesh, optional detection or
  initial pose depending on the protocol.
- Metrics: BOP Average Recall over VSD, MSSD, and MSPD; ADD or ADD-S where
  compatible with prior work.
- Purpose: compare with established object-pose methods.

### Task B: Initialized multi-view tracking

- Input: a ground-truth or benchmark-provided first-frame pose, followed by a
  sequence without pose ground truth at inference time.
- Metrics: translation and rotation error, relative pose error, success rate,
  time to first failure, longest failure run, and area under the success curve.
- Purpose: isolate tracking and multi-view evidence from detection quality.

### Task C: Automatic initialization and recovery

- Input: no privileged pose after sequence start; controlled view dropout,
  occlusion, or pose perturbation is injected.
- Metrics: recovery success rate, recovery latency, post-recovery pose error,
  false recovery triggers, and runtime.
- Purpose: test RQ3 directly.

## 4. Required Experimental Conditions

- monocular left;
- monocular right;
- legacy equal-weight fusion;
- legacy outlier rejection plus smoothing;
- joint multi-view scoring without refinement;
- joint scoring plus refinement;
- full reliability and recovery method;
- oracle view selection as an upper-bound diagnostic;
- one, two, and more-than-two views where the dataset permits;
- ground-truth masks versus predicted masks, reported separately;
- visibility and occlusion bins;
- slow, medium, and fast object-motion bins.

## 5. Metric Groups

### Absolute pose accuracy

- translation error in centimeters;
- rotation geodesic error in degrees;
- BOP Average Recall;
- ADD or ADD-S AUC where model symmetry permits.

### Sequence accuracy

- relative pose error at multiple temporal offsets;
- drift from the last reliable frame;
- trajectory success AUC under joint translation/rotation thresholds.

### Robustness and recovery

- failure rate;
- time to first failure;
- number and duration of failure episodes;
- recovery success and latency;
- performance under camera dropout and corrupted depth.

### Temporal behavior

- velocity and acceleration error relative to ground truth;
- prediction jerk as a secondary diagnostic only;
- lag introduced by temporal smoothing.

A low jitter score alone is not evidence of accuracy: a constant but incorrect
pose is perfectly smooth.

### Reliability calibration

- expected calibration error;
- negative log likelihood where covariance is predicted;
- risk-coverage curve and area under the risk-coverage curve;
- error versus predicted uncertainty by view and visibility bin.

### Efficiency

- wall-clock milliseconds per frame;
- peak GPU memory;
- preprocessing, hypothesis generation, joint scoring, refinement, and
  recovery time reported separately.

## 6. No-GT Evaluation on `clip03`

The following are valid proxy diagnostics:

1. **Held-out-view evaluation**: estimate a pose from one view or a subset of
   views, then evaluate silhouette, depth, and feature residuals in a withheld
   view. This is the strongest available no-GT test.
2. **Cross-view rendering consistency**: silhouette IoU or boundary F-score,
   visible-surface depth residual, and robust feature reprojection error.
3. **Stereo consistency**: bidirectional disparity consistency and agreement of
   object points lifted independently from different camera pairs.
4. **Temporal correspondence consistency**: compare predicted object motion
   with image feature tracks or optical flow on visible object regions.
5. **Controlled corruption tests**: inject view dropout, mask erosion,
   corrupted depth, or initial-pose perturbations and measure whether the
   system detects and recovers from the known intervention.
6. **Manual sparse audit**: annotate a small set of frames with qualitative
   alignment grades for failure analysis, without claiming metric GT.

Pseudo-ground-truth from bundle adjustment may be useful for debugging, but it
must be labelled as pseudo-GT and cannot be the sole evidence for the main
accuracy claim.

## 7. Evaluation RQs

| RQ | Hypothesis | Main experiment | Main metrics |
|---|---|---|---|
| RQ1 | Stereo foundation evidence improves geometry and robustness beyond independently generated depth maps. | Compare mono depth, one-way warped depth, bidirectional stereo, and multi-pair camera-graph evidence. | BOP AR, depth residual, performance by visibility. |
| RQ2 | Joint observation-level estimation outperforms late pose fusion. | Compare left/right, average, oracle selection, joint scoring, and joint refinement. | Pose error, success AUC, calibration. |
| RQ3 | Reliability-aware recovery prevents persistent drift. | Inject view failures and evaluate natural failure episodes. | Failure duration, recovery rate and latency. |

## 8. Official Resources

- HOT3D: <https://github.com/facebookresearch/hot3d>
- HOT3D paper: <https://arxiv.org/abs/2411.19167>
- BOP tasks and metrics: <https://bop.felk.cvut.cz/tasks/>
- BOP toolkit: <https://github.com/thodan/bop_toolkit>
- H2O paper: <https://arxiv.org/abs/2104.11181>
- DexYCB: <https://dex-ycb.github.io/>
- HO3D paper: <https://arxiv.org/abs/1907.01481>

