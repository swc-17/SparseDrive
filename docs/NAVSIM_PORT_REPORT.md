# SparseDrive on NAVSIM — project report

**Dates:** 2026-07-21 → 2026-07-23 (ongoing: combined training + final evals)
**Branch:** `navsim-port` (this repo). Companion branch:
`tejan/noiser-injection-harness` (a6afa98, branched from a508a8d) — holds the
noiser/oracle injection harness (`tools/eval_gt_planner_geometry.py`, oracle
hooks); those files are intentionally NOT in this branch's tree.
**Running log with full evidence:** `docs/NAVSIM_PORT_SESSION_STATUS.md`.
Runbooks: `docs/navsim_eval_openloop.md`, `docs/navsim_eval_closedloop.md`.
Dataset comparison: `docs/navsim_vs_nuscenes_comparison.md`.

## Objective

Port the original SparseDrive (detection + online map + motion + planning)
from nuScenes to NAVSIM with a geometry-only (V6) planner
(`geometric_inputs=True`, `use_cam_ego_feature=False`), train stage-1 →
stage-2 on navtrain, evaluate open- and closed-loop, and compare three
training-data regimes of the identical architecture:

- **A — NAVSIM-only** (navtrain, frozen 1,067-log train split)
- **B — NAVSIM + nuScenes combined** (~121k samples, 1:1 sample weighting)
- **C — nuScenes-only** (the unified-contract baseline reproduction)

## What was built (all committed on `navsim-port`)

1. **Data boundary**: NAVSIM→SparseDrive converter implementing the full
   audited contract (coordinate rotation, 7-class taxonomy, deterministic
   track ids, 8-step/4s ego futures, command masking, ego-status
   construction, connected-run sequences, 8-camera order + distortion);
   `NavSim3DDataset`; nuPlan-GPKG local vector-map extraction; per-view
   undistortion transform. Gated by calibration/map overlays (visually
   verified) and bit-level contract tests.
2. **navtrain production**: 103,288 scenarios converted (exact official
   count), frozen log-disjoint 90/10 split, all four k-means anchors from
   the train split, 179 GB sharded frame tars + infos + anchors on S3,
   resumable multi-node staging.
3. **Training infra**: single-node and 2-node/16-GPU Lilypad harnesses
   (placement groups, per-node staging barrier, torchrun rendezvous);
   sampler proven trajectory-identical across topologies.
4. **Unified nuScenes contract** for B/C: regenerated nuScenes infos
   (7-class + original labels, 8-step ego futures bit-matching the old
   6-step prefix, padded cameras with provably-zero attention contribution),
   `MultiSourceConcatDataset`, combined anchors.
5. **Evaluation stack**: 8-step/Pacifica `PlanningMetric`; devkit-free
   7-class detection mAP + map chamfer mAP; NAVSIM `AbstractAgent` wrapper
   (temporal resets per token, SD→NAVSIM inverse, SE(2) output); v1 PDMS and
   v2 two-stage EPDMS scoring with pinned caches (navtest cache generated:
   12,146/12,146; full navhard cache: 5,912/5,912); parameterized
   single-node and consolidated multi-eval cluster eval jobs; all
   reproducibility gates byte-identical (Human navtest 0.9455, CV 0.2065).

## Verified results to date

### Model A — NAVSIM-trained V6 (16-GPU chain, `navsim_stage2_geoinput_full_16g/iter_8703`)

| Metric | Value |
| --- | --- |
| navtest v1 PDMS (official open-loop, 12,146 tokens) | **0.7620** (NC 0.950, DAC 0.878, EP 0.724, TTC 0.873, comfort 1.000) |
| navmini PDMS (dev) | 0.8371 |
| navtrain-val L2 (0.5–4.0 s) | avg **0.792 m** (0.19 → 1.33 m); obj-box collision 0.64% avg |
| Stage-1 detection (navtrain-val, 10,435 samples) | mAP **0.3517**, NDS-analog 0.4103 |
| Stage-1 online map | chamfer mAP **0.8503** |
| navhard two-stage EPDMS | **0.1917** (stage-one 0.526, stage-two 0.346; 5,912 tokens, local run 2026-07-23) |
| Zero-shot nuScenes val L2 / collision | **1.125 m / 1.042%** (6-step comparable subset; 8-step avg 1.494 m, L2@4s 2.60 m — job vq71ao) |

Training cost: stage-1 13.6 h (2×8 A100, batch 128, lr 4e-4) + stage-2 3.5 h.
The single-node reference chain (batch 32, official-recipe schedule, 29.5 h)
scored navmini 0.8318 vs A's 0.8349 (cluster harness) and **navtest 0.7689
vs A's 0.7620** (local harness, 12,146 tokens) — deltas of ±0.003–0.007 in
opposite directions across splits: the 2.2x-faster batch-128 configuration
is practically equivalent to the official recipe.

### Model C — nuScenes-only unified V6 (`nusc_unified_stage2_geoinput_seed0/iter_5860`)

| Metric | Value |
| --- | --- |
| nuScenes val L2 avg / collision (identical subset + GT as baseline) | **0.656 m / 0.195%** vs baseline seeds 0.639–0.660 / 0.177–0.254 — **reproduction gate PASS** |
| L2@1/2/3 s deltas vs baseline mean | +0.004 / +0.009 / +0.018 m (pre-registered gate ≤0.022) |
| 8-step tail (unified only) | L2@3.5 s 1.269 m, @4.0 s 1.533 m |
| Zero-shot NAVSIM navmini PDMS | **0.4902** (EP 0.48, TTC 0.60 — cross-domain perception gap) |
| Zero-shot NAVSIM navtest PDMS | **0.4569** (12,146 tokens, local run 2026-07-24) |

The gate result also closed the 8-vs-6-step question with metrics: the
elevated training `planning_reg` loss is entirely the far-horizon averaging
effect; shared-horizon quality is at baseline.

### Model B — combined NAVSIM+nuScenes V6

Stage-1 training in progress (`sd_combined_stage1_2n-qmynkc`, 2×8 A100,
~24% at 17:00, ETA ~08:00 Jul 24), after two dataset-wrapper/train.py
contract bugs were found by failed launches, fixed, and validated by running
the real `tools/train.py` path locally (`73c25d3`, `9324025`). Stage-2 +
full dual-domain eval follow tomorrow.

### External comparison — SparseDriveV2 released checkpoints (same pinned caches)

| Model | navtest v1 PDMS | navhard 2-stage EPDMS |
| --- | --- | --- |
| SparseDriveV2 (planning-only, trajectory vocabulary + scorer) | **0.9222** (claim 92.22 confirmed to 4 dp) | **0.3753** / **0.4046** (bug-fix scorer) |
| SparseDrive-V1 port, Model A (full stack, geometry-only planner) | 0.7620 | **0.1917** (stage1 0.526 / stage2 0.346) |
| Human / constant velocity | 0.9455 / 0.2065 | — |

Notable: V2's published "90.3" is a **navtest**-v2 EPDMS number (reproduced:
0.9037); it was never evaluated on navhard by its authors and collapses
there (stage-two 0.500; lane-keeping/extended-comfort failures). Caveats
recorded: V2's ckpts trained on all 1,192 navtrain logs vs our 1,067
(~12% more data), and a V2 harness bug on synthetic tokens required a
documented monkey-patch (`scripts/sdv2_eval/`).


### Why B trails C on nuScenes — root-cause analysis (2026-07-24)

B's +11% L2 (0.731 vs 0.656) with collision parity was investigated on the
user's hypothesis that perception, not the geometry-only planner, carries
the domain difference. **Confirmed — B's nuScenes perception is degraded:**
det mAP **0.3498 vs C's 0.4687** (NDS 0.4294 vs 0.5438; mATE 0.616 vs
0.463 m; worst on nuScenes' low-count classes: bicycle 0.16 vs 0.35,
barrier 0.31 vs 0.49). Noisier box geometry feeds the V6 planner directly.
A second, independent factor is quantified on the planner's output side:
the combined plan anchors (k-means over a 77% NAVSIM pool; nuScenes is
only 5.6% of left-turn samples) leave nuScenes' fast/long turns uncovered
(worst left-turn mode 17.3 m from its nearest combined anchor). Both trace
to the same root: unweighted 1:1-per-sample mixing gives nuScenes a 4.3x
exposure deficit vs C at matched compute. Map-quality comparison (sd_evalcmp_bc_map2): **B's nuScenes mapping
collapsed — map mAP 0.199 vs C's 0.552** (ped_crossing 0.085 vs 0.495,
boundary 0.193 vs 0.579) — larger than the det gap and the dominant
suspect for the L2 regression (map polylines are direct V6 planner
inputs). fix_height offsets cover both ground planes (+-0.26 m), so the
ground_height=-1.1 scalar is unlikely the cause; leading hypotheses are
the 4.3x nuScenes exposure deficit and nuScenes-vs-GPKG map labeling
convention conflict. **DISCRIMINATOR RESOLVED (B-v2, 2026-07-25): exposure.** The 4x-upweighted
finetune fully restored nuScenes maps — map mAP 0.199 -> **0.5679, now
above C's 0.552** (ped_crossing 0.085 -> 0.531) — no labeling-convention
conflict. Detection recovered most of its gap (mAP 0.350 -> 0.4232 vs C
0.4687; mATE 0.616 -> 0.518 vs 0.463). Planning: collision 0.202% ->
**0.157% (beats C's 0.195%)**; L2 0.731 -> 0.722 (~10% above C persists,
consistent with the residual detection-precision gap). Candidate fixes: nuScenes-upweighted sampling,
per-dataset-balanced anchors; a stage-2-only retrain (~4 h) would test the
planner-side share with perception held fixed.

## Target deliverable: the A/B/C matrix (identical architecture, data-only axis)

| | A: NAVSIM-only | B: combined | C: nuScenes-only |
| --- | --- | --- | --- |
| NAVSIM navtest PDMS | 0.7620 | **0.7493** (B-v1) | zero-shot: **0.4569** |
| NAVSIM navhard EPDMS | **0.1917** | 0.1712 (B-v1: s1 0.499 / s2 0.311) | — |
| nuScenes val L2 / collision | zero-shot: **1.125 m / 1.042%** | v1: 0.731 m / 0.202% -> **v2: 0.722 m / 0.157%** (collision now BEATS C; L2 gap persists) | 0.656 m / 0.195% |
| navtrain-val det mAP / map mAP | 0.352 / 0.850 | pending | — |

## Failure ledger (all diagnosed and resolved)

1. 16g stage-1 NaN at iter ~2.4k (sqrt-scaled lr 5.7e-4) → restart at 4e-4.
2. Combined launches ×2: `MultiSourceConcatDataset` rejected train.py's
   injected `work_dir`, then lacked `CLASSES` → fixed + locally validated
   through first forward.
3. Cluster eval smoke: unstaged log pickles (scene_loader=0) → staging fix.
4. Cross-hardware PDMS repro gate breach (Δ0.0022) → per-token analysis:
   one near-boundary token flip; accepted with caveat.
5. Accidental duplicate stage-2 submission (submit.sh `PATTERN*` glob) →
   stopped, no data loss; pitfall recorded.
6. UNRESOLVED — cluster reaper kills eval workloads at ~45-90 min: every
   configuration tried was terminated (1-GPU; 8-GPU consolidated; verified
   GPU-busy keepalive; other.training model_identifier; 8-way sharded with
   all 8 GPUs in active inference). Long training jobs survive; sub-40-min
   eval jobs survive. Not diagnosable or automatable from this side —
   needs the infra owner. Local-GPU evals are the working path (blocked
   2026-07-25 by an NVML driver mismatch pending reboot).

## Git/branch state notes (as of this report)

- `navsim-port` history around the eval work was linearized in a parallel
  session: the two navsim-eval commits formerly known as b2ca809/930a9a8
  now exist as **ef4e0db/5ec4b47** (re-applied without an intervening noiser
  commit). Neither old hash was pushed; if any remote ever saw them, the
  next push needs care.
- The noiser injection harness lives exclusively on
  `tejan/noiser-injection-harness` (a6afa98). Future injection runs must
  check out that branch or cherry-pick a6afa98.
- Pre-existing uncommitted WIP (modified `lilypad_entrypoint.py`,
  `requirements_lilypad.txt`, `tools/save_planner_inputs_cjepa.py`;
  untracked `emperror/`, `geo_planner/`, probe tools, declutter docs) is
  deliberately untouched and uncommitted throughout.

## Next steps

1. Tonight: A navhard EPDMS + consolidated (ref navtest/navhard, C zero-shot
   navtest) → complete A column + reference verdict.
2. Jul 24: B stage-1 → stage-2 → consolidated dual eval (B full suite +
   A-on-nuScenes zero-shot) → complete matrix → final comparison section.
3. Open offers not yet taken: multi-node as default for future runs;
   image-feature planner ablation (non-V6 stage-2 config exists); noiser
   injection experiments on the trained A/B checkpoints (harness branch).
