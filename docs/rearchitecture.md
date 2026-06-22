# Project Rearchitecture

Last updated: 2026-06-14

## 1. Design Principle

The repository should be organized around a reproducible experiment graph:

```text
dataset -> observations -> hypotheses -> joint state estimate -> evaluation
```

Third-party models are adapters that produce observations or hypotheses. They
must not define project-wide paths, coordinate conventions, or result formats.

## 2. Target Architecture

```text
src/stereo_hoi/
  data/
    schema.py              canonical camera, observation, and sequence types
    adapters/              HOT3D, H2O, DexYCB, HO3D, legacy ZED
  geometry/
    se3.py                 transforms and geodesic operations
    camera.py              projection, unprojection, camera-rig transforms
  observations/
    stereo.py              stereo depth/correspondence/confidence adapter
    masks.py
    features.py
    rendering.py
  hypotheses/
    foundationpose.py      per-view proposal generation
    relocalization.py
  methods/
    legacy_fusion.py       frozen equal-weight baseline
    joint_scoring.py       shared-state cross-view scoring
    joint_refinement.py
    reliability.py
    recovery.py
  evaluation/
    pose.py
    trajectory.py
    calibration.py
    no_gt.py
    bop.py
  experiments/
    runner.py
    registry.py
  vis/
  cli.py
configs/
  dataset/
  method/
  experiment/
tests/
```

## 3. Canonical Contracts

### Coordinate frame

- Each pose name must encode direction, such as `rig_from_camera` or
  `camera_from_object`.
- Ambiguous names such as `T`, `pose`, and `extrinsics` should not cross module
  boundaries.
- The canonical estimated state is `rig_from_object`.
- Dataset adapters are responsible for converting native coordinates and units
  to meters and right-handed homogeneous transforms.

### Observation

Each view observation must carry:

- frame and camera identifiers;
- camera intrinsics and `rig_from_camera`;
- image size and optional RGB, depth, mask, or feature references;
- validity and visibility metadata;
- provenance, including which model and configuration produced it.

### Result

Every method emits:

- one `rig_from_object` pose per frame;
- uncertainty or confidence;
- active and rejected views;
- whether initialization or recovery occurred;
- component runtimes;
- a serialized experiment configuration hash.

## 4. Migration Stages

### Stage 0: Freeze legacy behavior

- Keep current `depth`, `tracking`, `fusion`, and `hand` commands operational.
- Label current fusion as `legacy_average` in experiments.
- Add regression tests for known transforms and fusion output.

### Stage 1: Evaluation foundation

- Add canonical schemas, SE(3) utilities, pose and trajectory metrics.
- Add a generic `evaluate` command for predicted and GT pose directories.
- Integrate official BOP toolkit for BOP metrics rather than duplicating it.

### Stage 2: Dataset adapters

- HOT3D first, using official toolkit and clipped sequences for development.
- H2O or DexYCB second to verify multi-view generalization.
- Legacy ZED adapter for `clip03`.

### Stage 3: Observation refactor

- Replace right-depth forward warp with bidirectional stereo inference or
  symmetric disparity conversion.
- Expose stereo confidence and consistency maps.
- Lift per-view object observations into the rig frame.

### Stage 4: Method refactor

- Generate per-view hypotheses.
- Score every hypothesis in all views.
- Refine one shared rig-frame pose.
- Predict or estimate view reliability.
- Add persistent-failure recovery.

### Stage 5: Experiment system

- Configuration-driven runs with immutable outputs.
- Dataset/method/seed/config recorded in every result directory.
- Automatic metric tables, visibility breakdowns, and failure reports.

## 5. Non-goals

- Do not vendor or rewrite FoundationStereo, FoundationPose, or WiLoR inside the
  core package.
- Do not introduce a learned fusion network before a measurable geometry-based
  baseline exists.
- Do not treat temporal smoothing as tracking accuracy.
- Do not use no-GT consistency as the headline metric.

