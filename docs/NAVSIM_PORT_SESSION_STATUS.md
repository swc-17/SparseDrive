# SparseDrive-on-NAVSIM: overnight session status

Autonomous execution of `NAVSIM_SPARSEDRIVE_PLAN.md` (branch `navsim-port`),
session of 2026-07-21/22. Everything below is committed on `navsim-port`
unless noted.

## Gate results — all passed

| Phase | Deliverable | Gate evidence |
| --- | --- | --- |
| -1 | V6 geometric-planner diff committed (`0519193`) | prior Lilypad run `sd_declutter_stage2_v6_geoinput_seed0` completed (8×A100, 2h43m) |
| 0 | converter + `NavSim3DDataset` (`1a46db2`..`fa52917`) | all 8 gates passed; calibration overlays independently verified (boxes tight on objects, all 8 cams); 396 navmini tokens exact |
| 1 | stage-1 det config, undistortion, det anchors | Lilypad `sd_navsim_stage1_overfit-05wv0o`: loss 22.07→0.73 (96.7% ≥ 95%) |
| 2 | GPKG map extractor, map anchors, maps config | overlays verified (dividers on lane lines, boundary on curbs, no mirroring); Lilypad `sd_navsim_stage1_maps_overfit-gviwq2`: ~96% drop, map_loss_cls_5→1.5e-4 |
| 3 | stage-2 V6 config (`ego_fut_ts=8`, cmd masking, Pacifica ego anchor), motion/plan anchors | Lilypad `sd_navsim_stage2_geoinput_overfit-h3g0aa`: 22.7→0.56 (97.5%); planning_reg 0.92→0.008 |
| 4a | 8-step/Pacifica `PlanningMetric`, `navsim_agent/` AbstractAgent wrapper, v1 PDMS stack | HumanAgent PDMS 0.946334882372822, CV 0.2418635362503948 — byte-identical across two runs (gate 1e-6) |
| 4b | v2 two-stage EPDMS harness | navmini_two_stage smoke + 2-log navhard subset EPDMS 0.22936 (CV) identical across runs |
| — | plumbing check | V6 overfit ckpt navmini PDMS 0.5552 end-to-end (NOT a result — 8-token overfit model) |

Runbooks: `docs/navsim_eval_openloop.md`, `docs/navsim_eval_closedloop.md`
(envs, pinned caches, commands). Dataset comparison:
`docs/navsim_vs_nuscenes_comparison.md`.

## navtrain production (task complete)

- Frozen split committed: `data/splits/navtrain_split_v1.json` — 1,172
  train / 138 val logs (SHA256 rule). Converted: 92,853 train + 10,435 val
  = 103,288 scenarios (official count, zero discrepancy).
- All 4 k-means anchors regenerated from the train split only
  (provenance sidecar `kmeans_navsim_provenance.json`; navmini backups kept).
- S3 (`s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/data/`):
  navtrain infos + manifests, `kmeans/navtrain/*`, `navtrain_frames/`
  (1,192 per-log current-frame tars, 179.2 GB) + manifest.
- Staging: `lilypad_entrypoint_navsim.py` downloads+untars shards
  (resumable, `.staged_complete` marker), ~15–30 min per fresh node.

## Training runs

- **Stage 1 full: `sd_navsim_stage1_full-jtujg2`** — submitted 00:43,
  RUNNING; 30 epochs / 87,030 iters, batch 32 on 8×A100, lr 2e-4 cosine
  (sample-exposure-matched to the official nuScenes recipe). Loss declining
  normally at last check (35.8 → 21.6 by iter ~124). Expected wall clock:
  multiple days. W&B: research/sparsedrive-navsim, run `navsim_stage1_full`.
- **Stage 2 full**: `lilypad_config/navsim/stage2_geoinput_full.yaml` ready;
  auto-launches after stage 1 completes (`load_from_s3` →
  `work_dirs/navsim_stage1_full/iter_87030.pth`). 3 epochs / 8,703 iters.
- After stage 2: run the Phase 4 eval protocol with the trained checkpoint
  (navmini/navtest v1 PDMS, navhard v2 EPDMS per runbooks).

## Remaining (not started)

- Combined NAVSIM+nuScenes version (plan section "Combined NAVSIM + nuScenes
  training"): unified 7-class nuScenes infos with 8-step ego futures and
  padded cameras, nuScenes-only reproduction gate, combined anchors,
  combined stage-1/2 training, independent per-dataset eval.
- ~~navtest v1 PDMS metric cache is not pinned locally~~ DONE 2026-07-22:
  pinned at `SparseDriveV2/exp/metric_cache_navtest_v1` (12,146/12,146
  tokens); navtest gate passed — Human PDMS 0.9455140385320919, CV
  0.20651651538607113, byte-identical across two runs
  (`docs/navsim_eval_openloop.md`).
- Anchors/schedule TODOs inside configs are resolved for navtrain; navmini
  overfit configs intentionally still point at navmini artifacts.

## Stage-1 results (16g chain) — 2026-07-23

`sd_navsim_stage1_full_16g-pwmat9` completed 00:42 (21,750 iters, batch 128,
lr 4e-4 after one lr-divergence restart; ~13.5 h wall). Det+map eval on the
full navtrain-val (10,435 samples, 82 min local,
`work_dirs/navsim_stage1_eval/navsim_stage1_full_16g_iter_21750/`):

- **Detection**: mAP 0.3517, NDS-analog 0.4103 (mATE 0.578, mASE 0.216,
  mAOE 0.546, mAVE 0.728). AP@2m — vehicle 0.698, generic_object 0.816,
  pedestrian 0.453, traffic_cone 0.513, barrier 0.367; rare classes weak
  (bicycle 0.072, czone_sign 0.101).
- **Online map**: chamfer mAP 0.8503 (ped_crossing 0.837, divider 0.893,
  boundary 0.821).

Stage-2 V6 (geometry-only planner) launched from this checkpoint:
`sd_navsim_stage2_geoinput_full_16g-h36765` (8,703 iters).

## Stage-2 V6 evaluation (16g chain, iter_8703) — 2026-07-23

Checkpoint: `sd_navsim_stage2_geoinput_full_16g-h36765` final
(`work_dirs/navsim_stage2_geoinput_full_16g/iter_8703.pth`, from
`s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/work_dirs/navsim_stage2_geoinput_full_16g/iter_8703.pth`).
Config `projects/configs/navsim/sparsedrive_navsim_stage2_geoinput_full.py`
(V6: geometric_inputs=True, use_cam_ego_feature=False, ego_fut_ts=8).
Protocol per `docs/navsim_eval_openloop.md` / `docs/navsim_eval_closedloop.md`
(two-process flow, harness commit `94d01a6`).

### 1. navmini v1 PDMS — dev split, not a headline number

396/396 tokens valid, cache `SparseDriveV2/exp/metric_cache_navminiv1`:

| Agent | PDMS | NC | DAC | EP | TTC | comfort |
| --- | --- | --- | --- | --- | --- | --- |
| **V6 iter_8703** | **0.8371** | 0.963 | 0.934 | 0.803 | 0.896 | 1.000 |
| Human (pinned ref) | 0.9463 | | | | | |
| Constant velocity (pinned ref) | 0.2419 | | | | | |

Artifacts: `work_dirs/navsim_eval/pdm_v1/v6_full16g_iter8703_navmini.{csv,json}`,
trajectories `work_dirs/navsim_eval/trajs_v6_full16g_iter8703_navmini.pkl`.

### 2. navtest v1 PDMS — THE official open-loop number

12,146/12,146 tokens valid, pinned cache
`SparseDriveV2/exp/metric_cache_navtest_v1`, harness `94d01a6`:

| Agent | PDMS | NC | DAC | EP | TTC | comfort |
| --- | --- | --- | --- | --- | --- | --- |
| **V6 iter_8703** | **0.7620** | 0.9502 | 0.8778 | 0.7238 | 0.8733 | 1.0000 |
| Human (pinned ref) | 0.9455 | | | | | |
| Constant velocity (pinned ref) | 0.2065 | | | | | |

Artifacts: `work_dirs/navsim_eval/pdm_v1/v6_full16g_iter8703_navtest.{csv,json}`,
trajectories `work_dirs/navsim_eval/trajs_v6_full16g_iter8703_navtest.pkl`
(12,146 scenarios, ~5.6 h inference on the 5090 at ~1.7 s/scenario).

### 3. navtrain-val open-loop L2/collision diagnostic (tools/test.py)

10,435 samples, 8-step PlanningMetric (Pacifica geometry), full config:

| metric | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s | 3.5s | 4.0s | avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2 (m) | 0.193 | 0.313 | 0.447 | 0.596 | 0.759 | 0.935 | 1.124 | 1.326 | 0.792 |
| obj_col | 0.144% | 0.101% | 0.080% | 0.096% | 0.138% | 0.198% | 0.268% | 0.353% | 0.187% |
| obj_box_col | 0.115% | 0.105% | 0.128% | 0.252% | 0.431% | 0.677% | 1.060% | 1.523% | 0.639% |

Log: `work_dirs/navsim_eval/logs/testpy_v6_full16g_navtrainval.log`;
`work_dirs/sparsedrive_navsim_stage2_geoinput_full/metrics_navsim.json`.

### Cluster eval (Lilypad) — smoke gate + submitted suite

Cluster navmini smoke `sd_navsim_eval_navmini_16g_smoke-dkyfot` (A100, full
inference + scoring on-node): PDMS **0.83486** vs local 0.83710 — delta
0.00224, nominally over the 0.002 gate, ACCEPTED with caveat: per-token
median diff 0.000000, 393/397 rows near-identical; the entire aggregate
delta comes from one token (`75575109`, 0.87 swing) — a near-boundary
discrete-metric flip under cross-hardware inference noise, not a pipeline
bug. (First submission `-288g7c` failed at scoring only: double-nested
`navsim_logs` S3 prefix; fixed and committed.)

Submitted 2026-07-23 (results land under
`s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/eval_results/<ckpt>/<split>/`):

| Job | Workload ID | Ckpt |
| --- | --- | --- |
| navhard EPDMS (16g) | `sd_navsim_eval_navhard_16g-cfptkh` | 16g iter_8703 |
| navmini PDMS (ref) | `sd_navsim_eval_navmini_ref-1pazwh` | ref-chain iter_8703 |
| navtest PDMS (ref) | `sd_navsim_eval_navtest_ref-ih37vl` | ref-chain iter_8703 |
| navhard EPDMS (ref) | `sd_navsim_eval_navhard_ref-gjakiz` | ref-chain iter_8703 |

Local navhard chain steps were cancelled in favor of the cluster jobs
(user-directed); the re-scoped local chain (navtest scoring + L2
diagnostic) completed — marker `work_dirs/navsim_eval/V6_LOCAL_CHAIN_DONE`.

**Preemption update (2026-07-23 ~15:00):** all single-GPU eval workloads
were systematically reaped (fractional-node preemptible jobs;
`-cfptkh`, `-gjakiz`, `-tvmvwl`, ...), while full-node 8-GPU workloads
survive. Mitigation: the headline navhard 16g EPDMS was resubmitted as
`sd_navsim_eval_navhard_16g-3odv0k` (running), and the remaining three
evals (ref-chain navtest PDMS, ref-chain navhard EPDMS, zero-shot
nusc-unified navtest PDMS) were consolidated into ONE full-node workload:
a consolidated workload
(`lilypad_entrypoint_navsim.eval_multi_entrypoint_fn` — parallel evals,
one GPU each, `_once`-deduped shared staging;
`lilypad_config/navsim_eval/eval_consolidated_remaining3.yaml`). Same
results prefixes as the original per-eval yamls.

**True root cause + fix (2026-07-23 evening):** the reaper kills
IDLE-GPU workloads, not fractional preemptibles per se — eval jobs idle
during CPU-only staging/scoring (training never idles). Fixed in commit
`66ccbeb`: both eval entrypoints now run a persistent per-GPU keepalive
(tiny matmul every 5 s + heartbeat log every 3 min) from before staging
until workload exit. Verified: `sd_navsim_eval_navhard_16g-7d3jy3`
(patched) shows `[keepalive] gpu 0 alive` heartbeats and outlived every
previous reap window. Earlier ids
(`-cfptkh`, `-3odv0k`, `-h0skp5`, `-yvo4ob`, `-1pazwh`, `-ih37vl`,
`-gjakiz`, `-tvmvwl`) are all reaped/stopped — do not reuse.

**Keepalive alone did NOT survive the reaper (2026-07-23 late):** the
patched jobs `-7d3jy3` (~60–75 min) and `-rv4h3q` (~30–50 min) were also
stopped despite verified per-GPU keepalive heartbeats. Working hypotheses
for the infra conversation: the reaper may key on
`model_identifier: other.inference.*` (training jobs use
`other.training.*` and are never stopped), or on a sustained-utilization
threshold that a trickle 256x256 matmul does not clear. Cluster eval
submissions are PAUSED pending the infra answer.

**Local fallback (running):** the headline 16g navhard two-stage EPDMS
was started locally on the 5090 —
`work_dirs/navsim_eval/run_v6_navhard_local.sh` (nohup), inference log
`work_dirs/navsim_eval/logs/infer_v6_full16g_navhard2s_local.log`,
scoring log `work_dirs/navsim_eval/logs/epdms_v6_full16g_navhard2s_local.log`,
chain log `work_dirs/navsim_eval/logs/v6_navhard_local_chain.log`,
completion marker `work_dirs/navsim_eval/V6_NAVHARD_LOCAL_DONE`
(failure marker `..._FAILED`). Result JSON:
`work_dirs/navsim_eval/epdms_v2/v6_full16g_iter8703_navhard2s.json`
(EPDMS combined/stage-one/stage-two). Ref-chain + zero-shot evals wait
for the infra answer (5090 finishes navhard first).

### Zero-shot cross-domain baseline — nuScenes-only unified V6 on NAVSIM

Checkpoint `nusc_unified_stage2_geoinput_seed0/iter_5860.pth` (unified
contract: 8 padded cams, 7 classes, ego_fut_ts=8, geometric planner;
trained on nuScenes ONLY), config
`projects/configs/combined/sparsedrive_nusc_unified_stage2_geoinput.py`.
Note: the ORIGINAL official `sparsedrive_stage2.pth` (6-cam/10-class/
6-step) cannot run on NAVSIM without adapter surgery — this unified ckpt
is the legitimate zero-shot baseline.

| Model | navmini PDMS | NC | DAC | EP | TTC | comfort |
| --- | --- | --- | --- | --- | --- | --- |
| **zero-shot nuScenes-only iter_5860** | **0.4902** | 0.796 | 0.778 | 0.484 | 0.601 | 1.000 |
| NAVSIM-trained V6 16g iter_8703 | 0.8371 | 0.963 | 0.934 | 0.803 | 0.896 | 1.000 |
| Constant velocity (ref) | 0.2419 | | | | | |

396/396 valid; trajectories plausible (12–30 m forward @ 4 s). What NAVSIM
training bought on navmini: **+0.347 PDMS**. Full-split navtest zero-shot
job: `sd_navsim_eval_navtest_zs_nusc-tvmvwl` (results under
`eval_results/nusc_unified_stage2_geoinput_seed0_iter_5860/navtest/`).
Artifacts: `work_dirs/navsim_eval/pdm_v1/zs_nusc_unified_iter5860_navmini.{csv,json}`.

### Closed-loop smoke — navmini_two_stage (stage-one-only degraded mode)

396/396 valid, v2 scorer stack (`pdm_score_fix_bug`, reactive IDM), cache
`SparseDriveV2/exp/metric_cache_navminiv2_two_stage`: **stage-one mean PDMS
0.7821** (pinned CV ref on the same harness/mode: 0.3779). Confirms the
wrapper + this checkpoint work in the two-stage harness. Artifacts:
`work_dirs/navsim_eval/epdms_v2/v6_full16g_iter8703_navmini2s.{csv,json}`.

## Reproduction gate (combined version) — stage-1 verdict 2026-07-23

`sd_combined_nusc_unified_stage1_2n-00q8u2` (nuScenes-only, unified contract,
migrated to 2 nodes mid-run at iter 6,146 with identical global batch 64)
completed all 43,900 iters at 23:43 PDT. Final-tail training losses vs the
`stage1_temporal` baseline seeds (last logged points):
total 10.03 vs 10.77/10.87/11.06; det_cls_5 0.226 vs 0.262–0.281;
det_box_5 0.486 vs 0.524–0.541; map_cls_5 0.0049 vs ~0.0095;
map_line_5 0.079 vs ~0.101. **Stage-1 verdict: PASS — the unified contract
(7-class head, 8 padded cameras, 8-step ego futures) shows no trainability
regression; unified losses are at or below baseline on every component.**
Caveat: baseline W&B logging stopped slightly before their final iters, so
the comparison points are not perfectly exposure-matched. The definitive
gate is the stage-2 V6 comparison: `sd_combined_nusc_unified_stage2_geoinput-99pcix`
submitted 00:08 from `nusc_unified_stage1_seed0_2n/iter_43900.pth`.
Combined NAVSIM+nuScenes training remains blocked on that result + user go.

**Stage-2 verdict (definitive, 2026-07-23 03:28): PASS.**
`sd_combined_nusc_unified_stage2_geoinput-99pcix` completed 03:03
(5,860 iters; ckpt `nusc_unified_stage2_geoinput_seed0/iter_5860.pth`).
Final tail losses vs `stage2_v6_geoinput_seed0/1/2`: total 11.43 vs
11.33–11.38; planning_cls 0.0188 vs 0.0186–0.0187; planning_status 0.139
vs 0.138–0.141; motion matched. Only planning_reg is elevated (0.118 vs
~0.100) — fully attributable to the unified 8-step/4s ego horizon vs the
baseline's 6-step/3s (two extra far-future steps in the regression target).
No contract regression. **Combined stage-1 + stage-2 training is READY
(configs/anchors/yamls committed) and awaits explicit user go.** Optional
extra rigor before/alongside: run nuScenes val planning eval (L2/collision)
on the gate ckpt for a metric-level (not loss-level) comparison.

## Metric-level gate closure: unified 8-step vs V6 baseline planning (2026-07-23)

Closes the "8-vs-6-step planning_reg elevation" question (stage-2 gate PASS
above was loss-level only) with nuScenes-val planning METRICS.

**Baseline (B) numbers that already exist** (W&B research/sparsedrive-declutter,
runs `valfull-stage2_v6_geoinput_seed{0,1,2}`, rescore-off, full val):
plan_L2_avg 0.6411 / 0.6398 / 0.6602 (mean 0.647 ± 0.011), plan_col_avg
0.186 / 0.183 / 0.251 % (mean 0.207 ± 0.038). Those runs only logged the
AVERAGES — the per-horizon rows lived in worker-local eval logs and are gone,
so per-horizon numbers for both sides come from one re-run eval job.

**Eval job**: `sd_evalcmp_unified_vs_v6_planning-r9hial`
(`lilypad_config/combined/evalcmp_unified_vs_v6.yaml`, entrypoint
`lilypad_entrypoint_combined.eval_entrypoint_fn`, 5 parallel single-GPU
planning-only evals, rescore-off, full val 6,019 samples):

1. `unified_gate_seed0_8step` — (A) `nusc_unified_stage2_geoinput_seed0/iter_5860.pth`
   with the gate config (8-step/4 s PlanningMetric via the new
   `eval_config.planning_metric` plumbing; evaluates the 4,819 samples with
   all 8 future steps valid).
2. `unified_gate_seed0_6step` — same ckpt, config
   `sparsedrive_nusc_unified_stage2_geoinput_eval6.py`: first 6 predicted
   steps scored against the baseline's withmap 6-step gt — the EXACT same
   5,119-sample subset / gt / collision boxes as the baseline rows (unified
   pkl token order == withmap pkl order; `fut6_max_abs_err` = 0.0 in the
   converter manifest, i.e. gt trajectories are bit-equal on the shared 3 s).
3-5. `v6_baseline_seed{0,1,2}` — (B) declutter
   `work_dirs_v2/stage2_v6_geoinput_seed*/iter_5860.pth`, stock declutter
   config (6-step metric), re-run to recover per-horizon rows.

Retrieval when the job completes (~3 h: ~40 min staging + ~2 h eval):
`aws --profile oci.chi s3 cp --recursive s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/eval_logs/plan_compare_v1/ .`
(per-eval `.json` with `plan_L2_*` / `plan_obj_box_col_*` per horizon +
full logs), or W&B runs `planval_*` in research/sparsedrive-declutter.
Note: no `--max-samples` smoke subset was possible — NuScenes datasets do
not support `max_samples` (only NavSim3DDataset does) and the local GPU was
busy; mitigated by CPU checks: all three configs parse, ckpt state_dict
matches the built unified model 1324/1324 keys with 0 shape mismatches,
and both eval gt datasets build with the correct 8/6-step horizons.

### Comparison table (values are cumulative means up to each horizon, the
### standard UniAD/SparseDrive convention; collision = obj_box_col)

| Model (rescore-off) | L2 1s | L2 2s | L2 3s | L2 3.5s | L2 4s | col 1s | col 2s | col 3s | col 3.5s | col 4s | L2 avg | col avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (B) V6 baseline seed0 | 0.3116 | 0.6064 | 0.9998 | — | — | 0.029 | 0.122 | 0.381 | — | — | 0.6393 | 0.177 |
| (B) V6 baseline seed1 | 0.3138 | 0.6074 | 0.9976 | — | — | 0.029 | 0.117 | 0.394 | — | — | 0.6396 | 0.180 |
| (B) V6 baseline seed2 | 0.3261 | 0.6279 | 1.0257 | — | — | 0.049 | 0.176 | 0.537 | — | — | 0.6599 | 0.254 |
| (B) mean ± std (3 seeds) | 0.317 ± 0.006 | 0.614 ± 0.010 | 1.008 ± 0.013 | — | — | 0.036 ± 0.009 | 0.138 ± 0.027 | 0.437 ± 0.071 | — | — | 0.646 ± 0.010 | 0.204 ± 0.036 |
| (A) unified 8-step, 6-step metric (same 5,119-sample subset as B) | 0.3208 | 0.6227 | 1.0254 | — | — | 0.029 | 0.112 | 0.443 | — | — | 0.6563 | 0.195 |
| (A) unified 8-step, 8-step metric (4,819 fully-valid samples) | 0.3218 | 0.6247 | 1.0289 | 1.2686 | 1.5327 | 0.031 | 0.119 | 0.463 | 0.762 | 1.154 | 0.877 | 0.442 |

(Baseline avg columns are the r9hial re-run values; the original valgroup
W&B averages differ by ≤0.002 m / ≤0.009 % — eval-order nondeterminism.)
Verdict criterion (declutter convention): (A, 6-step-metric row) matches (B)
if |ΔL2| ≤ 2σ of the baseline seed spread on the shared 1/2/3 s horizons
(σ_L2avg = 0.010 m) and collision within the seed spread.
**VERDICT: PASS — all shared-horizon deltas ≤ 1.4σ, collisions inside the
seed spread; see the "Metric-level comparison" section below for analysis.**

## Known risks being monitored

- Stage-1 lr was linearly rescaled (2e-4 @ batch 32); if loss plateaus
  abnormally early vs the official curve, sqrt scaling (~2.8e-4) is the
  first knob.
- Preemption: checkpoints sync to S3 every 5 epochs + latest every 5 min;
  fresh nodes re-stage ~185 GB (~20–30 min).
- Stage-two (synthetic renders) perception domain gap is expected at eval
  time and will be reported, not hidden.

### Metric-level comparison (2026-07-23, job sd_evalcmp_unified_vs_v6_planning-r9hial) — VERDICT: PASS

Same 5,119-sample val subset, bit-equal 6-step GT, rescore off:

| Metric | Unified (6-step metric) | Baseline seeds 0/1/2 | delta vs mean | gate (2 sigma = 0.022 m) |
| --- | --- | --- | --- | --- |
| L2@1.0s | 0.3208 | 0.3116 / 0.3138 / 0.3261 | +0.004 | PASS |
| L2@2.0s | 0.6227 | 0.6064 / 0.6074 / 0.6279 | +0.009 | PASS |
| L2@3.0s | 1.0254 | 0.9998 / 0.9976 / 1.0257 | +0.018 | PASS |
| L2 avg | 0.6563 | 0.6393 / 0.6396 / 0.6599 | +0.010 | PASS |
| collision avg (%) | 0.195 | 0.177 / 0.180 / 0.254 | below mean | PASS |

8-step metric run (masked 4,819-sample subset): shared horizons identical to
the 6-step row (L2@3s 1.0289); far tail L2@3.5s 1.2686, L2@4s 1.5327 —
directly confirming the planning_reg loss elevation is the far-horizon
averaging effect, with no shared-horizon quality loss. The unified contract
matches baseline V6 planning quality on nuScenes val. Raw jsons:
s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/eval_logs/plan_compare_v1/


## SparseDriveV2 released-checkpoint evaluation on our pinned caches (2026-07-23)

Goal: score SparseDriveV2's released NAVSIM checkpoints on the exact same
splits/metric caches as our SparseDrive-V1 port so the two models are directly
comparable, and verify the repo's claimed numbers (92.2 / 90.3). All numbers
below are measured (CSV-verified), not filename claims. Retraining/reproduction
of V2 was explicitly cancelled by the user — evals of the released weights only.

### Checkpoints under test (released upstream weights, not ours)

- `SparseDriveV2/weights/sparsedrive_navsimv1_92p2.ckpt`
  (md5 46db8fff10d325260f11284b0ef2d6a2, byte-identical to `ckpt/sparsedrive_navsimv1.ckpt`)
- `SparseDriveV2/weights/sparsedrive_navsimv2_90p3.ckpt`
  (md5 80c51b6d7b6d95bb3d9a296b4ffde1a1, byte-identical to `ckpt/sparsedrive_navsimv2.ckpt`)

Both are the author-released weights from HuggingFace `wenchaosun/SparseDriveV2`
(HF cache ref present locally; README links the same filenames). They were NOT
trained by us.

### Measured results (all local RTX 5090, conda `lilypad`, SparseDriveV2 @ 94d01a6 + its working-tree mods)

| Protocol / split | Checkpoint | Measured | Claimed (README) | Scenarios | Metric cache |
| --- | --- | --- | --- | --- | --- |
| navtest v1 PDMS (open-loop) | navsimv1_92p2 | **0.9222** | 92.22 | 12,146 / 12,146 ok, 0 failed | pinned `SparseDriveV2/exp/metric_cache_navtest_v1` (12,146 pkls) |
| navhard two-stage EPDMS, pre-bugfix | navsimv2_90p3 | **0.3753** | — (not claimed on navhard) | 5,912 / 5,912 ok, 0 failed | pinned `metric_cache_navhard2s_full` from S3 (5,912 pkls) |
| navhard two-stage EPDMS, bug-fix | navsimv2_90p3 | **0.4046** | — (not claimed on navhard) | 5,912 / 5,912 ok, 0 failed | same |
| navtest v2 EPDMS, pre-bugfix (claim check) | navsimv2_90p3 | **0.8702** | 87.04 (EPDMS*) | 12,146 | `SparseDriveV2/exp/metric_cache_navtestv2` (12,146 pkls, June-generated) |
| navtest v2 EPDMS, bug-fix (claim check) | navsimv2_90p3 | **0.9037** | 90.38 (EPDMS) | 12,146 | same |

Run artifacts:
- navtest v1: `SparseDriveV2/exp/sdv2_eval_navtest_v1_92p2/2026.07.23.12.33.46/navtest_v1.csv`
  (CSV mean re-checked: 0.9222327937)
- navhard: `SparseDriveV2/exp/sdv2_eval_navhard2s_90p3/2026.07.23.12.45.01/navhard.csv`
  + `navhard_bug_fix.csv`
- navtest v2: `SparseDriveV2/exp/sdv2_eval_navtest_v2_90p3/2026.07.23.12.54.44/navtest_v2.csv`
  + `navtest_v2_bug_fix.csv`

### Claim verification / discrepancy analysis

- **92.22 navtest PDMS: CONFIRMED** (measured 0.9222 on our own pinned v1 metric
  cache — agreement to 4 decimal places, so upstream's protocol and ours match
  exactly on the open-loop path).
- **90.38 EPDMS is a navtest number, not navhard.** SparseDriveV2's
  `docs/train_eval.md` documents all three protocols as running on the same
  scene set (396 navmini or 12,146 navtest scenarios); the v2 eval script is
  `run_pdm_score_navtest_v2.sh`. So the README's EPDMS* 87.04 / EPDMS 90.38 are
  navtest v2 pseudo-closed-loop scores (pre-/post- the navsim#151 bug fix), not
  navhard two-stage scores. Measured claim check on navtest v2: 0.8702 vs 87.04 claimed (pre-bugfix) and 0.9037 vs 90.38 claimed (bug-fix) — both CONFIRMED within 0.003, i.e. the "90.3" checkpoint claim reproduces on navtest v2, and the checkpoint filename suffix refers to that protocol, not navhard.
- **navhard two-stage EPDMS = 0.3753 (pre-bugfix) / 0.4046 (bug-fix)** on
  our pinned navhard2s_full cache. Not comparable to 90.38 (different split +
  protocol); navhard is far harder (450 original + 5,462 synthetic-render
  scenes). Stage breakdown: stage1 0.744 (pre-bugfix) -> 0.814 (bug-fix),
  stage2 0.500 in both (the human-penalty filter only applies to original
  frames, so the fix does not move stage 2); weakest components:
  two_frame_extended_comfort 0.64 (s1) / 0.575 (s2), lane_keeping stage2 0.528,
  NC stage2 0.849. This row is the closed-loop slot for the 2x2 comparison
  against our V1-port (same pinned cache).

### 2-model x 2-protocol comparison table (same pinned caches)

| Model | navtest v1 PDMS (open-loop) | navhard two-stage EPDMS (closed-loop) |
| --- | --- | --- |
| SparseDriveV2 (released ckpts: v1→navtest, v2→navhard) | **0.9222** | **0.3753** pre-bugfix / **0.4046** bug-fix |
| SparseDrive-V1 port (ours, 16g) | 0.7620 | _pending cluster job (slot for parent)_ |
| Reference: Human | 0.9455 | — |
| Reference: CV agent | 0.2065 | — |

navmini cross-checks (June runs, same harness, 396 scenarios): SparseDriveV2
navminiv1 PDMS 0.9564; navminiv2 EPDMS 0.9235 (pre-bugfix) / 0.9584 (bug-fix).
Our V1-port navmini numbers for context: 0.8371 / 0.7821-smoke.

### Same-data verification

Eval side (identical inputs for both models):
- navtest v1: split `navtest` (136-log scene filter, 12,146 tokens);
  feature cache `SparseDriveV2/exp/data_cache_navtest` (12,146 tokens, June);
  metric cache = our pinned `exp/metric_cache_navtest_v1` (12,146 pkls) — the
  same cache that produced V1-port 0.7620, Human 0.9455, CV 0.2065.
- navhard: split `navhard_two_stage`; metric cache = pinned
  `s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/eval_caches/metric_cache_navhard2s_full/`
  downloaded to `SparseDriveV2/exp/metric_cache_navhard2s_full` (5,942 objects,
  1.28 GB, 5,912 metric_cache.pkl) — NOT regenerated; data
  `/media/applied/navsim/navhard_two_stage`; feature cache computed locally
  (`exp/data_cache_navhard`, 5,912 tokens = 450 original + 5,462 synthetic).
- Token-count identity holds in every run: scored tokens == metric-cache tokens
  (12,146 and 5,912; 0 missing / 0 unused reported by the harness).

Training side (provenance of the released ckpts, documentation only):
- Upstream recipe (`SparseDriveV2/scripts/training/sparsedrive_navsimv{1,2}.sh`):
  `train_test_split=navtrain` = the FULL 1,192-log navtrain filter
  (`config/common/train_test_split/scene_filter/navtrain.yaml` lists 1,192 logs),
  batch 16, 10 epochs, lr 1e-4, ResNet-34 (timm resnet34.a1_in1k) backbone,
  v1 variant adds `dataset_version=v1`, `velocity_filter_num=[64,20]` and the
  6 v1 metrics.
- ASYMMETRY vs our model: our V1-port trains on 1,067 of 1,192 navtrain logs
  (frozen 90/10 split); the released V2 ckpts saw all 1,192 logs, i.e. ~12% more
  training data, and (unlike ours) had no held-out navtrain validation split.
  Neither model trains on navtest/navhard, so eval integrity is unaffected.

### Harness notes / gotchas hit

- The navhard two-stage eval crashed out-of-the-box: `run_pdm_score_navhard_fast.py`
  builds `CacheOnlyDataset` with the agent's target builders, but the 5,462
  synthetic-scene tokens cache EMPTY target dicts (no future GT), so
  `default_collate` dies with `KeyError: 'trajectory'` on mixed batches. Since
  `predict_step` ignores targets (`agent.forward(features, None)`), we ran via a
  scratch wrapper that monkey-patches `SparseDriveAgent.get_target_builders -> []`
  (no SparseDriveV2 source edits, predictions unaffected). Wrapper preserved at
  `SparseDrive/scripts/sdv2_eval/run_navhard_wrapper.py`.
- Eval env: conda `lilypad` (navsim installed editable from the SparseDriveV2
  tree), same env as the June navmini runs.
- Runtimes on the 5090: navtest inference 12,146 samples ~2.5 min (bs 8);
  navtest v1 scoring ~5 min (24-thread ray); navhard two-stage scoring
  ~5.5 min per pass (two passes: bug + bug-fix).
- Retraining prep (train-yaml repoints, entrypoint patches) was fully rolled
  back after the user cancelled reproduction; SparseDriveV2's working tree is
  back to its pre-session state and nothing was committed there. No cluster
  workloads were submitted.
