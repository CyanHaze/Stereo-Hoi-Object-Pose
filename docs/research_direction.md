# Research Direction

Last updated: 2026-06-14

## 1. Project Positioning

This project should no longer be positioned as "stereo FoundationPose plus
pose averaging." That formulation describes an engineering baseline, but it
does not define a defensible research contribution.

The proposed research problem is:

> Given synchronized and calibrated stereo or multi-view video, a known object
> mesh, and object observations under severe hand occlusion, estimate one
> accurate, temporally stable, and recoverable 6D object trajectory.

The intended paper type is a **technique paper**. Public benchmarks and a
sequence-level evaluation protocol support the method; they are not intended
to become a separate benchmark-paper contribution in the first project cycle.

## 2. Core Scientific Claim

The project should test the following claim:

> Multi-view information is most useful when fused as object-centric
> observations before or during state estimation, rather than by averaging
> independently estimated final poses.

The method should maintain a single latent object state for each time step.
Stereo depth, pairwise matching evidence, masks, rendered geometry, and
view-specific pose hypotheses are measurements of that state. They are not
independent final answers with manually assigned equal weights.

## 3. Research Questions

### RQ1: Stereo foundation evidence and multi-view generalization

How can the correspondence and geometry learned by a stereo foundation model
be used as object-level measurement evidence, and how can this pairwise stereo
evidence be generalized naturally to a calibrated camera graph with more than
two views?

This question starts from the defining strength of FoundationStereo rather
than from pose averaging. Candidate signals include:

- bidirectional disparity or depth;
- left-right consistency;
- pairwise matching or correlation features;
- occlusion and invalid-region estimates;
- object-conditioned point or feature evidence lifted into a common rig frame.

For more than two cameras, each stereo pair becomes an edge in a camera graph.
Pairwise evidence should be aggregated in an object-centric representation,
while avoiding double-counting correlated observations.

### RQ2: Observation-level multi-view state estimation

How can RGB, mask, depth, correspondence, and rendering evidence from all
available views be used to estimate one shared 6D object state with calibrated
view-dependent uncertainty?

The target formulation is a joint estimator:

\[
\hat{T}_t = \arg\min_T \sum_v \rho\left(
r_{t,v}(T)^\top \Sigma_{t,v}^{-1}r_{t,v}(T)\right)
+ \lambda E_{\mathrm{temporal}}(T, T_{t-1}),
\]

where `r` contains observable residuals, `Sigma` represents uncertainty, and
`rho` is a robust loss. A first implementation can use FoundationPose to
generate hypotheses, but hypotheses must be scored and refined jointly across
views.

### RQ3: Failure detection and recovery in long sequences

How can the tracker detect view-specific degradation or persistent drift and
recover using valid views, temporal history, or global re-localization?

The current `clip03` failure at frame `00938` motivates this question: the
right-view tracker drifts for the remaining 293 frames. Rejecting that view
prevents contamination but does not recover its tracking state.

## 4. Method Hypothesis

The first method version should contain three matched modules:

1. **Stereo/Multi-view Evidence Encoder**
   Produces per-view geometry and confidence from calibrated image pairs. It
   should replace the current one-way left-depth-to-right forward warp.
2. **Object-centric Joint Hypothesis Scoring and Refinement**
   Transforms pose hypotheses to a common rig frame, renders each hypothesis in
   every available view, and optimizes one shared object pose using silhouette,
   depth, correspondence, and feature residuals.
3. **Reliability and Recovery Manager**
   Detects inconsistent or uninformative views, tracks uncertainty over time,
   and triggers cross-view or global re-localization after persistent failure.

The implementation may begin without training a new network. A geometry-based
joint estimator with measurable residuals is a valid first research baseline.
A learned reliability model is justified only after the residual-based system
and its failure modes are established.

## 5. Paper Logic Skeleton

| Stage | Content |
|---|---|
| Research background | Known-object 6D tracking is a critical front end for world-space HOI reconstruction. Hand occlusion makes a single view brittle, while synchronized wearable or fixed multi-camera systems provide complementary evidence. |
| Limitation 1 | Existing project code estimates final poses independently and combines them using manually selected rules, losing image- and geometry-level evidence. |
| Limitation 2 | The current right-view depth is synthesized by one-way warping from the left depth, so the two tracking branches are not independent and right-only regions are poorly observed. |
| Limitation 3 | Current tracking has no uncertainty calibration, persistent-failure recovery, or sequence-level evaluation. |
| Key idea | Represent stereo or multi-view outputs as uncertain object observations and jointly estimate one recoverable object trajectory in a common rig frame. |
| Challenge 1 | Pairwise stereo evidence must be made symmetric, confidence-aware, and extensible to a camera graph. |
| Challenge 2 | Heterogeneous residuals from multiple views must be combined without equal-weight assumptions or correlated-evidence overcounting. |
| Challenge 3 | A failed view must be detected and reinitialized without destabilizing valid views. |
| Module A | Stereo/Multi-view Evidence Encoder. |
| Module B | Object-centric Joint Hypothesis Scoring and Refinement. |
| Module C | Reliability and Recovery Manager. |
| Contribution 1 | A stereo-to-multi-view observation formulation for known-object tracking. |
| Contribution 2 | A joint, uncertainty-aware multi-view pose estimator. |
| Contribution 3 | A sequence-level evaluation and failure-recovery study on public benchmarks and real stereo video. |

Self-consistency checks:

- Limitations -> key idea: pass.
- Key idea -> challenges: pass.
- Challenges -> modules: pass.
- Modules -> contributions: pass.

## 6. Scope Control

### In scope

- rigid known-object 6D pose tracking;
- calibrated stereo and multi-view cameras;
- severe hand occlusion;
- long-sequence drift detection and recovery;
- public benchmark evaluation;
- hand reconstruction as optional observation or downstream validation.

### Out of scope for the first paper

- a new hand reconstruction model;
- full world-space human or body reconstruction;
- unknown object mesh reconstruction;
- a newly collected large benchmark;
- a complete replacement for AGILE, WHOLE, or other HOI systems.

## 7. Decision Record

- `clip03` remains a qualitative and cross-device test sequence because it has
  no metric pose ground truth.
- HOT3D is the primary benchmark candidate.
- Public benchmark ground truth replaces the need to create ground truth for
  the main method study, but it does not provide ground truth for `clip03`.
- Existing equal-weight fusion remains only as a legacy baseline.
- The next implementation target is joint cross-view hypothesis scoring, not a
  more complicated hand-written pose averaging rule.

