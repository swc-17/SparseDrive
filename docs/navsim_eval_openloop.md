# NAVSIM open-loop evaluation (Phase 4a)

Two complementary open-loop measurements over the 4 s / eight-step horizon
(plan: `NAVSIM_SPARSEDRIVE_PLAN.md`, "Evaluation design"):

1. **L2 / collision diagnostic** — SparseDrive's `PlanningMetric` extended to
   8 steps at 0.5 s with Pacifica ego geometry; runs inside `tools/test.py`.
2. **NAVSIM v1 PDMS** — the official open-loop-with-simulation score, produced
   with the pinned v1 metric cache inside SparseDriveV2's environment.

## Environments (pinned)

| Piece | Environment | Notes |
| --- | --- | --- |
| SparseDrive model forward | conda `sparsedrive310` (`/home/tejan/miniconda3/envs/sparsedrive310`) | mmcv 1.7.2 / mmdet 2.28.2 / torch 2.8.0, GPU |
| PDMS scoring | conda `lilypad` (`/home/tejan/miniconda3/envs/lilypad`) | `navsim` 2.0.0 installed editable from `/home/tejan/lwm-rl/SparseDriveV2` (commit `94d01a68cc710f358c2e130d693c983d1e177e49`, dirty user checkout — only committed code is relied on), nuplan-devkit, torch 2.8.0 |

No local env has both stacks, so SparseDrive PDMS uses a **two-process flow**:
inference writes a `{token: (8,3) NAVSIM SE(2) poses}` pickle
(`navsim_agent/run_inference_infos.py`, sparsedrive310), scoring consumes it
(`navsim_agent/score_pdm_v1.py --agent traj:<pickle>`, lilypad). The
single-process `AbstractAgent` wrapper (`navsim_agent/agent.py`, registered
via `navsim_agent/config/sparsedrive_original_agent.yaml` mirroring
SparseDriveV2's hydra agent pattern) is the target interface once a combined
env exists; it shares all code (adapter/runner/coord) with the two-process
flow.

The v1 metric-cache pickles reference `navsim.navsim_v1.*` classes — they are
scored ONLY inside SparseDriveV2's env. Never mix them with a DrivoR-native
cache or with v2 EPDMS caches.

## 1. L2/collision diagnostic (tools/test.py)

Config: `projects/configs/navsim/sparsedrive_navsim_stage2_geoinput.py` —
`eval_config.planning_metric = dict(n_future=8, ego_width=2.297,
ego_length=5.176, ego_height=1.777, ego_center_offset=1.461)` (Pacifica,
rear-axle frame). nuScenes configs keep bit-identical defaults
(6 steps, 1.85/4.084, 0.5 m UniAD offset).

```bash
cd /home/tejan/lwm-rl/SparseDrive
PYTHONPATH=$PWD /home/tejan/miniconda3/envs/sparsedrive310/bin/python \
    tools/test.py projects/configs/navsim/sparsedrive_navsim_stage2_geoinput.py \
    <checkpoint.pth> --eval planning
```

Verified wiring:
- feeding GT trajectories as predictions gives L2 = 0.0000 at every horizon
  (obj_col — the GT-collision diagnostic column — is 0.17% avg, as expected
  for GT boxes);
- a random-init checkpoint runs end-to-end and prints the 0.5–4.0 s table
  (log: `work_dirs/navsim_eval/logs/test_py_randominit.log`).

Note: `tools/test.py` streams navmini rows in converter order with streaming
temporal banks (same convention as the nuScenes diagnostic). The official
PDMS path below instead resets banks and replays the 4-frame history per
scenario token, as NAVSIM requires.

## 2. NAVSIM v1 PDMS

Pinned inputs:

- metric cache: `/home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navminiv1`
  (396 navmini tokens, complete);
- split: `navmini` (396 scenario tokens, 62 logs), scene filter from
  SparseDriveV2's `train_test_split/navmini.yaml`;
- scoring params: `default_run_pdm_score_fast_v1.yaml` /
  `default_scoring_parameters_v1.yaml` (PDMSimulator + v1 PDMScorer,
  40x0.1 s proposal sampling);
- harness: SparseDriveV2 commit `94d01a68cc710f358c2e130d693c983d1e177e49`;
- rescore: SparseDrive `use_rescore=False` (config), no heuristic rescoring.

### Reproducibility gate (PASSED 2026-07-21)

Pinned trivial agents, run twice each, deterministic sorted-token loop:

```bash
cd /home/tejan/lwm-rl/SparseDrive
for run in 1 2; do
  /home/tejan/miniconda3/envs/lilypad/bin/python navsim_agent/score_pdm_v1.py \
      --agent human            --run-tag human_run$run
  /home/tejan/miniconda3/envs/lilypad/bin/python navsim_agent/score_pdm_v1.py \
      --agent constant_velocity --run-tag cv_run$run
done
```

| Agent | Run 1 PDMS | Run 2 PDMS | max abs per-token diff | 396 tokens valid |
| --- | --- | --- | --- | --- |
| HumanAgent (pinned) | 0.946334882372822 | 0.946334882372822 | 0.0 (CSV byte-identical) | yes |
| ConstantVelocityAgent (pinned) | 0.2418635362503948 | 0.2418635362503948 | 0.0 (CSV byte-identical) | yes |

Gate threshold 1e-6; observed 0.0. Per-token CSVs + JSON metadata:
`work_dirs/navsim_eval/pdm_v1/{human,cv}_run{1,2}.{csv,json}`.

### navtest cache + reproducibility gate (PASSED 2026-07-22)

Full navtest v1 metric cache, generated and consumed inside SparseDriveV2's
tree/env (never mixed with DrivoR-native pickles):

- cache: `/home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navtest_v1`
  (**12,146 / 12,146** navtest tokens, 136 logs, 0 failures, 3.1 GB; token
  set exactly matches `train_test_split/scene_filter/navtest.yaml`);
- generated with SparseDriveV2's own
  `navsim/planning/script/run_metric_caching_v1.py`
  (`train_test_split=navtest`, default `ray_distributed_no_torch` worker,
  24 threads, ~14 min wall) at harness commit `94d01a6`;
- LiDAR note: the audit's 1,528 navtest tokens without current-frame LiDAR
  do **not** affect v1 metric caching or the trivial-agent gate — both load
  scenes with `SensorConfig.build_no_sensors()` (annotations + maps only),
  and all 12,146 tokens cached and scored successfully.

```bash
cd /home/tejan/lwm-rl/SparseDriveV2
OPENSCENE_DATA_ROOT=/media/applied/navsim NUPLAN_MAPS_ROOT=/media/applied/navsim/maps \
NAVSIM_EXP_ROOT=/home/tejan/lwm-rl/SparseDriveV2/exp \
/home/tejan/miniconda3/envs/lilypad/bin/python navsim/planning/script/run_metric_caching_v1.py \
    train_test_split=navtest \
    cache.cache_path=/home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navtest_v1

cd /home/tejan/lwm-rl/SparseDrive
for run in 1 2; do
  /home/tejan/miniconda3/envs/lilypad/bin/python navsim_agent/score_pdm_v1.py \
      --agent human --split navtest \
      --metric-cache /home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navtest_v1 \
      --run-tag human_navtest_run$run
  /home/tejan/miniconda3/envs/lilypad/bin/python navsim_agent/score_pdm_v1.py \
      --agent constant_velocity --split navtest \
      --metric-cache /home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navtest_v1 \
      --run-tag cv_navtest_run$run
done
```

Pinned navtest references (12,146/12,146 tokens valid in every run):

| Agent | Run 1 PDMS | Run 2 PDMS | max abs per-token diff |
| --- | --- | --- | --- |
| HumanAgent (pinned) | 0.9455140385320919 | 0.9455140385320919 | 0.0 (CSV byte-identical) |
| ConstantVelocityAgent (pinned) | 0.20651651538607113 | 0.20651651538607113 | 0.0 (CSV byte-identical) |

Gate threshold 1e-6; observed 0.0. Per-token CSVs + JSON metadata:
`work_dirs/navsim_eval/pdm_v1/{human,cv}_navtest_run{1,2}.{csv,json}`;
caching log `work_dirs/navsim_eval/logs/cache_navtest_v1.log`.

### SparseDrive V6 (geometric-planner) PDMS — plumbing check only

Checkpoint: Lilypad job `sd_navsim_stage2_geoinput_overfit-h3g0aa`,
`s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/work_dirs/navsim_stage2_geoinput_overfit/latest.pth`
(meta iter **1499**, snapshot pulled 2026-07-21 23:24 local, job still
running), local copy
`work_dirs/navsim_stage2_geoinput_overfit/latest_iter_1500.pth`.

This checkpoint deliberately **overfits eight frozen navmini tokens**
(Phase-3 gate manifest); its navmini PDMS is a plumbing check that the
inference -> coordinate-inverse -> heading -> PDMS pipeline runs end to end.
It is NOT a model result and must not be quoted as one.

```bash
# inference (sparsedrive310, GPU, ~12 min for 396 tokens)
/home/tejan/miniconda3/envs/sparsedrive310/bin/python -m navsim_agent.run_inference_infos \
    --config projects/configs/navsim/sparsedrive_navsim_stage2_geoinput.py \
    --checkpoint work_dirs/navsim_stage2_geoinput_overfit/latest_iter_1500.pth \
    --infos data/infos/navsim_infos_navmini.pkl \
    --output work_dirs/navsim_eval/trajs_v6_overfit_iter1500_navmini.pkl
# scoring (lilypad)
/home/tejan/miniconda3/envs/lilypad/bin/python navsim_agent/score_pdm_v1.py \
    --agent traj:work_dirs/navsim_eval/trajs_v6_overfit_iter1500_navmini.pkl \
    --run-tag v6_overfit_iter1500
```

| Checkpoint | Split | PDMS | Label |
| --- | --- | --- | --- |
| V6 stage-2 overfit, iter 1499 | navmini (396) | 0.5552384611971051 | plumbing check (8-token overfit model, NOT a result) |

For reference on the same cache/harness: constant velocity 0.2419, human
0.9463. Artifacts: `work_dirs/navsim_eval/pdm_v1/v6_overfit_iter1500.{csv,json}`,
trajectories `work_dirs/navsim_eval/trajs_v6_overfit_iter1500_navmini.pkl`.

Adapter correctness signal: on the eight frozen overfit tokens the
wrapper's 4-frame replay reproduces the GT plan with **0.06 m mean L2**
(per token: 0.13/0.04/0.03/0.05/0.03/0.16/0.01/0.03), vs ~5.9 m on unseen
navmini tokens — the replay path matches training semantics and the
SD->NAVSIM inverse is correct.

### Combined B-v2 EMPERROR supervision gate

The gate for `combined_stage2_v2_seed0/iter_8800.pth` exposes two independent
choices: `geometry_source=teacher` re-encodes the live selected top-50 object
anchors and top-10 map vectors, while `ego_query_only=true` removes the
query-side motion agents after fail-closed checks for the combined B-v2
geometry-only, `use_rescore=False` planner. The dedicated cluster config is
`lilypad_config/navsim_eval/eval_navmini_bv2_teacher_geometry.yaml`.

Local full NAVSIM-mini validation (396 tokens, 1,584 replay frames):

| Geometry source | PDMS | valid tokens | max encoder parity error |
| --- | ---: | ---: | ---: |
| native, ego-query-only | 0.7991659434 | 396/396 | n/a |
| teacher gate, ego-query-only | 0.7991864018 | 396/396 | 4.59e-6 |

Teacher minus native aggregate PDMS is `+2.05e-5`. Collision, drivable-area,
TTC, comfort, and driving-direction components are identical; only ego
progress changes. Median per-token maximum trajectory delta is 1.6 mm and
mean final-xy L2 is 1.32 cm. A few mode-sensitive scenes amplify floating-point
re-encoding noise (worst final-xy delta 1.12 m), so this establishes PDMS
parity, not bit-exact trajectory replay. Artifacts are
`work_dirs/navsim_eval/{trajs_bv2_{native_ego,teacher_geometry}_navmini.pkl,
pdm_v1/bv2_{native_ego,teacher_geometry}_navmini.{csv,json}}`.

#### Fixed untrained-EMPERROR negative control

A seed-0 randomly initialized `EmperrorCVAE` was sampled once per replay-frame
token from its prior, conditioned on that frame's genuine GT object/map sets.
Its selected `[50,11]` detections and `[10,40]` maps replaced the teacher
geometry through the same fail-closed gate. The trained B-v2 V6 checkpoint,
route command, ego state/history, and plan anchors remain active, so this is a
random-geometry negative control rather than a no-information random planner.

| Metric | Mean over 396 valid tokens |
| --- | ---: |
| PDMS | 0.2861878191 |
| no-at-fault collision | 0.6691919192 |
| drivable-area compliance | 0.4873737374 |
| ego progress | 0.2790802670 |
| TTC within bound | 0.5000000000 |
| comfort | 0.9873737374 |
| driving-direction compliance | 0.7916666667 |

All 396 scenarios and all 1,584 history frames used generated geometry;
`max_source_delta=62.49` confirms that the gate did not fall back to teacher
geometry. Exact GT-map reconstruction is pinned to Shapely `1.8.5.post1`
(396/396 current-frame maps match the stored annotations). The immutable
geometry artifact is
`s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/emperror/random_init_prior_sample/navmini_seed0.pkl`
with SHA-256
`f399651bca4b668b1de6867ec7d8c4735f02b8cfc8eb1c4fcdb58ad98f07b4c3`.
The cluster manifest is
`lilypad_config/navsim_eval/eval_navmini_bv2_untrained_emperror.yaml`; local
outputs are
`work_dirs/navsim_eval/trajs_bv2_emperror_random_seed0_navmini_exact.pkl` and
`work_dirs/navsim_eval/pdm_v1/bv2_emperror_random_seed0_navmini_exact.{csv,json}`.

### Reset / order-independence check (model inference)

With the V6 overfit checkpoint on GPU, predicting token A back-to-back
differs by up to ~8e-6 m (CUDA kernel nondeterminism — the deformable
aggregation op reduces with atomics); predicting A after other tokens
differs by ~1.2e-5 m, i.e. the same kernel-noise level. There is no
systematic temporal-state leak between scenario tokens; per-token bank
resets in `runner.predict_scenario` are effective. Consequence for
reproducibility: SparseDrive PDMS runs must score a **saved trajectory
pickle** (the two-process flow does exactly this) — scoring a fixed pickle
is bit-exact, while regenerating trajectories reproduces xy only to ~1e-5.

## Known eval-time caveats

- The `AgentInput` path zero-fills acc-z, angular rates, vel-z and steering
  in the 10-slot ego status (NAVSIM exposes only [vx, vy, ax, ay] there);
  the infos-based path (`run_inference_infos.py`) uses the converter's full
  can_bus status. vx (slot 6, the `InstanceQueue` contract) is present in
  both.
- `AgentInput` provides SE(2)-relative history poses; the synthetic
  `ego2global` used for temporal-bank projection assumes zero roll/pitch.
  The infos path uses the converter's full SE(3) poses.
- navtest v1 cache is pinned at
  `/home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navtest_v1` (see the
  navtest gate section above); navmini numbers are development-only.
