# Decision: adopt the SparseDriveV1.5 architecture as the stage-1 planner

**Date:** 2026-07-30. **Status:** ACCEPTED (Tejan, 2026-07-29).
**Scope:** SparseDriveV1.5 (`navsim_stage2_vocab_metric_16g/iter_8703`, branch
`research/sd15-v2-vocab`) becomes the stage-1 checkpoint for the three-stage
pipeline (stage 2: perception-error module; stage 3: closed-loop RL).
All numbers below are from the pinned evaluation stack (harness commit
`94d01a6`, pinned metric caches, byte-identical reproducibility gates);
per-token artifacts in `work_dirs/navsim_eval/`, aggregates in
`docs/abc_matrix_metrics_report.md`, W&B `research/sparsedrive-navsim`.

## What SparseDriveV1.5 is

V1 NAVSIM stage-1 perception (backbone + detection + online map) with a
V2-style trajectory-vocabulary planner (released 1024 paths x 256 velocity
profiles) whose candidate updates attend to the V6 agent/map **geometry
tokens** — no deformable image reads, no front-cam ego CNN — plus eight
EPDMS metric heads (BCE) used to re-rank the vocabulary at decode
(+0.016 PDMS / +0.018 EPDMS over pure imitation ranking). Trained on the
frozen 1,067-log navtrain split, same data as every in-house comparison row.

## The case

1. **Best in-house planner on every NAVSIM protocol.**

   | Model (planner) | navtest v1 PDMS ↑ | navtest v2 EPDMS ↑ | navhard EPDMS ↑ (s1/s2) |
   | --- | --- | --- | --- |
   | SparseDrive-Geo (regression, geometry-only) | 0.7620 | 0.7634 | 0.1917 (0.526/0.346) |
   | SparseDrive (regression, image features) | 0.8060 | 0.8051 | 0.1977 (0.581/0.347) |
   | **SparseDriveV1.5 (vocabulary + metric heads)** | **0.8613** | **0.8581** | **0.3297 (0.687/0.471)** |
   | SparseDriveV2 (released; diff arch, +12% data) | 0.9222 | 0.9037 | 0.4046 |

   +0.10 PDMS over the geometry regression planner and +0.055 over the
   image-feature regression planner, on identical perception and identical
   training data. Closed-loop (navhard) the margin widens: 0.330 vs 0.192/0.198
   — a 1.7x advantage where errors compound.

2. **The planner needs no image features — measured, not assumed.**
   The imgfeat ablation (identical twin of the geometry model, only the
   planner input changes) shows image features buy +0.055 stage-one navhard
   score on real frames (0.581 vs 0.526) and exactly nothing on synthetic
   renders (stage-two 0.347 vs 0.346). Stage-3 closed-loop training runs on
   simulated/noised inputs, where the image-feature advantage is measured to
   be zero. A geometry-interfaced planner gives up nothing in that regime.

3. **The geometry interface is exactly what the perception-error module
   needs.** The stage-2 module takes GT detections and predicts outputs in
   perception-error space (boxes/polylines + confidences). V1.5's planner
   consumes precisely those tokens (verified: `trajectory_vocab_planner.py`
   — "conditioned only on V6 geometry"), so noised GT slots directly into
   the planner with no feature fabrication.

4. **Metric heads are a native hook for stage 3.** The eight EPDMS BCE heads
   already predict per-candidate metric outcomes; closed-loop RL can
   finetune exactly these value-like heads and the re-ranking rule rather
   than bolting a critic onto a regression planner.

5. **Full perception stack retained.** Unlike released SparseDriveV2
   (planning-only, ResNet-34), V1.5 keeps the V1 detection/map heads — the
   same checkpoint family produces the GT↔detector-output export pairs the
   error module trains on.

## Risks and mitigations

- **Perception erosion in the vocab stage-2 finetune:** navtrain-val det
  mAP 0.2503 / map 0.7514 vs the regression twins' 0.34/0.84. The planner
  is measured to be robust to degraded perception inputs (the combined
  models plan within 0.013 PDMS of specialist with 4x-worse maps), and
  stage-2 exports can come from the stage-1 (pre-vocab-finetune) weights;
  freezing perception in future V1.5 finetunes is the structural fix.
- **navhard number is single-run** (same pinned cache/scorer as all rows,
  but without the repeat-run byte-identical pass the other rows got).
- **Gap to released SparseDriveV2 remains** (0.861 vs 0.922 navtest; 0.330
  vs 0.405 navhard). Known asymmetries: V2 trained on all 1,192 navtrain
  logs (~12% more) with no held-out split, and uses image reads its own
  way. Closing this gap is not a prerequisite for stages 2–3.

## Rejected alternatives

- **SparseDrive-Geo (Model A)** — same geometry interface, but 0.10 PDMS /
  0.14 navhard EPDMS behind V1.5 with no compensating advantage.
- **SparseDrive (imgfeat)** — best regression planner, but its edge exists
  only on real imagery (nothing on renders), and its planner would require
  the error module to fabricate instance image features.
- **B-v1/B-v2 (combined-data variants)** — resolved a different question
  (cross-domain transfer: detection transfers positively, the shared map
  head serves one map convention at a time, exposure picks which). Both
  trail on NAVSIM planning; the mapping convention conflict makes them the
  wrong base for a single-domain pipeline.
- **Released SparseDriveV2 checkpoint** — highest scores, but planning-only
  (no perception stack to export from), different training data, and
  external weights we cannot retrain under the same contract.

## Evidence trail

- Matrix + key messages: `docs/abc_matrix_metrics_report.md`
- Port provenance: `docs/NAVSIM_PORT_REPORT.md`
- Protocol runbooks: `docs/navsim_eval_openloop.md`, `docs/navsim_eval_closedloop.md`
- Scorers (traj-pickle based, gate-validated): `navsim_agent/score_pdm_v1.py`,
  `navsim_agent/score_epdms_navtest_v2.py`, `navsim_agent/score_epdms_two_stage.py`
- W&B: `research/sparsedrive-navsim`, group `abc_matrix_eval`
- Presentation: `docs/slides_abc_intern_20260729.html` (published artifact)
