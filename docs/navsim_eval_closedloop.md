# NAVSIM closed-loop evaluation (Phase 4b): two-stage EPDMS (v2)

NAVSIM's closed-loop protocol is the two-stage pseudo-closed-loop **EPDMS**
on `navhard`: stage one scores real scenes; stage two re-runs the agent from
its own stage-one endpoints inside synthetic (Gaussian-splat-rendered)
scenes. Harness: SparseDriveV2's
`navsim/planning/script/run_pdm_score_navhard_fast.py` machinery with
`train_test_split/navhard_two_stage.yaml`
(commit `94d01a68cc710f358c2e130d693c983d1e177e49`).

Entry point here: `navsim_agent/score_epdms_two_stage.py` (conda `lilypad`,
SparseDriveV2's eval env). It reuses the harness's scoring worker
(`run_pdm_score`) and aggregation (`create_scene_aggregators`,
`compute_final_scores`, `calculate_individual_mapping_scores`) verbatim and
only replaces trajectory production: the shipped harness requires
feature-cache agents (Lightning `CacheOnlyDataset`), which pinned trivial
agents and the original-SparseDrive wrapper are not. Scorer:
`pdm_score_fix_bug` (the harness's `navhard_bug_fix.csv` variant),
`worker=sequential`.

Data (all local):
- `/media/applied/navsim/navhard_two_stage/` — `openscene_meta_datas`,
  synthetic `sensor_blobs`, 5,462 `synthetic_scene_pickles`;
- maps `/media/applied/navsim/maps` (`NUPLAN_MAPS_ROOT`), logs
  `/media/applied/navsim/navsim_logs/test`.

## Smoke 1: navmini_two_stage (stage-one only, PASSED 2026-07-21)

`navmini_two_stage` as shipped in SparseDriveV2 contains **no**
`reactive_all_mapping` and **no synthetic scenes** (mini logs have no
synthetic renders — those exist only for navhard/test logs). It therefore
exercises the two-stage scorer stack (v2 scorer config, 16-future-frame
filter, reactive IDM traffic policy) on original frames only; the script
degrades gracefully to per-token stage-one PDMS and labels the run
accordingly (the shipped harness main() would crash on the missing mapping).

```bash
cd /home/tejan/lwm-rl/SparseDrive
/home/tejan/miniconda3/envs/lilypad/bin/python navsim_agent/score_epdms_two_stage.py \
    --agent constant_velocity --split navmini_two_stage \
    --metric-cache /home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navminiv2_two_stage \
    --run-tag cv_navmini2s_run1   # and _run2
```

Pinned cache: `SparseDriveV2/exp/metric_cache_navminiv2_two_stage`
(396 original navmini tokens).

| Agent | Run 1 stage-one mean PDMS | Run 2 | identical |
| --- | --- | --- | --- |
| ConstantVelocityAgent | 0.37790400683866376 | 0.37790400683866376 | yes — 396/396 valid, per-token diff 0.0, CSV byte-identical |

Artifacts: `work_dirs/navsim_eval/epdms_v2/cv_navmini2s_run{1,2}.{csv,json}`.

## Smoke 2 + reproducibility gate: genuine two-stage EPDMS on a navhard subset (PASSED 2026-07-21)

No navhard (synthetic-token) metric cache existed locally, so one was
generated for a two-log subset with SparseDriveV2's own `cache_data`
(fresh directory; the pinned full-split caches are never touched):

```bash
/home/tejan/miniconda3/envs/lilypad/bin/python navsim_agent/cache_metrics_two_stage_subset.py \
    --logs 2021.05.25.14.16.10_veh-35_01100_01664 2021.05.25.14.16.10_veh-35_01690_02183 \
    --output /home/tejan/lwm-rl/SparseDrive/work_dirs/navsim_eval/metric_cache_navhard2s_subset
```

Cache: 48 tokens = 4 original (stage one) + 44 synthetic (stage two).

```bash
for run in 1 2; do
  /home/tejan/miniconda3/envs/lilypad/bin/python navsim_agent/score_epdms_two_stage.py \
      --agent constant_velocity --split navhard_two_stage \
      --metric-cache /home/tejan/lwm-rl/SparseDrive/work_dirs/navsim_eval/metric_cache_navhard2s_subset \
      --logs 2021.05.25.14.16.10_veh-35_01100_01664 2021.05.25.14.16.10_veh-35_01690_02183 \
      --run-tag cv_navhard_subset_run$run
done
```

This runs the full two-stage path end to end: synthetic-scene loading,
stage-one + stage-two scoring with reactive IDM traffic, scene aggregation
(`reactive_all_mapping` filtered to the subset), two-frame extended comfort,
and EPDMS composition.

| Agent | Run | EPDMS combined | EPDMS stage one | EPDMS stage two |
| --- | --- | --- | --- | --- |
| ConstantVelocityAgent | 1 | 0.22936230474596875 | 0.5 | 0.4190076676954533 |
| ConstantVelocityAgent | 2 | 0.22936230474596875 | 0.5 | 0.4190076676954533 |

Gate: identical EPDMS across two runs — max abs per-token diff **0.0** over
all 48 rows (all valid). Artifacts:
`work_dirs/navsim_eval/epdms_v2/cv_navhard_subset_run{1,2}.{csv,json}`.

## Full navhard EPDMS (pipeline built 2026-07-23)

Concrete three-step flow (first exercised with the trained V6 16g
checkpoint; artifacts under `work_dirs/navsim_eval/`):

1. **Metric cache** — generated with SparseDriveV2's own
   `run_metric_caching.py` (v2 config, default ray worker) to a NEW
   directory, never touching the pinned caches:

   ```bash
   cd /home/tejan/lwm-rl/SparseDriveV2
   OPENSCENE_DATA_ROOT=/media/applied/navsim \
   NUPLAN_MAPS_ROOT=/media/applied/navsim/maps \
   NAVSIM_EXP_ROOT=/home/tejan/lwm-rl/SparseDriveV2/exp \
   /home/tejan/miniconda3/envs/lilypad/bin/python \
       navsim/planning/script/run_metric_caching.py \
       train_test_split=navhard_two_stage \
       metric_cache_path=/home/tejan/lwm-rl/SparseDrive/work_dirs/navsim_eval/metric_cache_navhard2s_full
   ```

   Result: `work_dirs/navsim_eval/metric_cache_navhard2s_full` — **5,912 /
   5,912** tokens (450 original stage-one + 5,462 synthetic stage-two),
   0 failures, 1.3 GB, ~6 min on 24 threads
   (log `work_dirs/navsim_eval/logs/cache_navhard2s_full.log`).

2. **Trajectories for ALL tokens** — two-process (no combined env):

   ```bash
   # (lilypad) dump per-token pre-pipeline frames: path-only cameras
   # (NAVSIM_CACHE_SKIP_IMAGES=1), committed AgentInput adapter semantics
   # (SE(2) history poses, zero-filled ego-status slots), LiDAR never
   # loaded (SparseDrive is camera-only; some test logs miss pcds)
   python navsim_agent/dump_frames_two_stage.py \
       --split navhard_two_stage \
       --metric-cache work_dirs/navsim_eval/metric_cache_navhard2s_full \
       --output work_dirs/navsim_eval/frames_navhard2s.pkl
   # (sparsedrive310, GPU) same bank-reset + history-replay runner as the
   # open-loop flow; synthetic stage-two renders go through the identical
   # undistort/resize pipeline as real frames
   python -m navsim_agent.run_inference_frames \
       --config projects/configs/navsim/sparsedrive_navsim_stage2_geoinput_full.py \
       --checkpoint <ckpt.pth> \
       --frames work_dirs/navsim_eval/frames_navhard2s.pkl \
       --output work_dirs/navsim_eval/trajs_<tag>_navhard2s.pkl
   ```

   Throughput ~2.0 s/scenario on the 5090 → ~3.3 h for 5,912 tokens.

3. **Score** with `score_epdms_two_stage.py --agent traj:<pickle>` (no
   `--logs`). At full scale pass `--worker ray_distributed_no_torch`
   (~7.5 s/token sequential ≈ 12 h vs well under an hour on 24 threads;
   the final CSV is token-sorted, so results are worker-order
   independent). Report EPDMS with harness commit, checkpoint, and token
   count pinned; keep v1 PDMS and v2 EPDMS in separate tables.

## Expectations / risks for stage two

- **Synthetic-render domain gap**: the V6 planner is geometry-only, but
  detection/map perception still runs on the Gaussian-splat renders.
  Expect degraded perception on stage-two frames; report the stage-two
  drop rather than hiding it (compare stage-one vs stage-two sub-scores in
  the CSV).
- Stage-two `AgentInput`s come from synthetic scene pickles; ego status
  there is the same [vx, vy, ax, ay] reduction noted in the open-loop
  runbook.
