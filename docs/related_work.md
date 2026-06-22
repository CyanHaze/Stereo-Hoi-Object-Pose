# Related Work Map

Last updated: 2026-06-14

This document complements the broader notes in
`F:/Research/02_Projects/HOI/HOI_survey.md`. It focuses on work directly
needed to position calibrated multi-view object tracking under hand occlusion.

## 1. Stereo Foundation Geometry

### FoundationStereo

FoundationStereo targets zero-shot stereo depth and introduces a foundation
stereo model designed to generalize across domains. Its relevance is not only
the final depth map: pairwise correspondence, geometry encoding, and stereo
consistency can become measurement evidence for object tracking.

- Project: <https://nvlabs.github.io/FoundationStereo/>
- Code: <https://github.com/NVlabs/FoundationStereo>
- Paper: <https://arxiv.org/abs/2501.09898>

### Fast-FoundationStereo

Fast-FoundationStereo makes the stereo foundation pipeline practical for
video-scale experiments. It is an efficiency baseline and an implementation
vehicle, not by itself the research contribution.

- Code: <https://github.com/NVlabs/Fast-FoundationStereo>
- Paper: <https://arxiv.org/abs/2509.17323>

## 2. Model-based 6D Object Pose

### FoundationPose

FoundationPose provides model-based pose estimation and tracking for novel
objects from RGB-D observations. In this project it should be treated as a
hypothesis generator and single-view baseline. Its final per-view poses should
not define the multi-view fusion abstraction.

- Project: <https://nvlabs.github.io/FoundationPose/>
- Paper: <https://arxiv.org/abs/2312.08344>

### BOP ecosystem

BOP supplies standardized object-pose datasets, symmetry-aware metrics, result
formats, and evaluation tools. The project should reuse BOP metrics rather than
creating incompatible pose-accuracy definitions.

- Tasks: <https://bop.felk.cvut.cz/tasks/>
- Toolkit: <https://github.com/thodan/bop_toolkit>

## 3. Multi-view Pose and Joint Consistency

Multi-view 6D pose work typically shares one principle relevant here: camera
observations should constrain one object state in a common frame. Candidate
mechanisms include joint hypothesis scoring, multi-view rendering residuals,
feature alignment, and bundle-style optimization.

The novelty claim must therefore not be "we use two cameras." It should be the
specific use of stereo-foundation evidence, uncertainty, and failure recovery
for long HOI sequences.

HOT3D already includes a multi-view object-pose baseline and shows the value of
multiple views. It is both the primary benchmark and evidence that naive
stereo extension is no longer a sufficient contribution.

## 4. HOI Reconstruction Context

The survey identifies two adjacent streams:

- camera/world-space hand reconstruction: HaWoR, Dyn-HaMR, and WiLoR;
- world-space HOI reconstruction: EgoGrasp, WHOLE, and Follow My Hold.

These methods motivate accurate object trajectories as an upstream component.
They should not all become direct baselines unless their task assumptions and
code availability match ours.

The first paper should evaluate whether the proposed object tracker improves a
downstream HOI pipeline only after the object-pose contribution is established.

## 5. Benchmark Datasets

- HOT3D: primary egocentric multi-view benchmark with mocap-grade annotations.
- H2O: synchronized egocentric and exocentric RGB-D HOI sequences.
- DexYCB: controlled calibrated multi-view hand-object pose benchmark.
- HO3D: established single-view hand-occlusion stress test.
- ARCTIC: future articulated-object extension, outside first-paper scope.

## 6. Literature Gaps to Track

The literature review remains incomplete in three areas and should be updated
before paper writing:

1. methods that expose uncertainty from stereo matching or cost volumes;
2. multi-view 6D pose methods with joint feature or rendering optimization;
3. long-term object trackers with explicit re-detection and recovery metrics.

Each future paper entry should record task assumptions, input modalities,
whether the object model is known, initialization protocol, multi-view fusion
stage, uncertainty mechanism, recovery mechanism, datasets, and metrics.

