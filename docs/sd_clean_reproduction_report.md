# sd-clean reproduction report

Purpose: prove the cleaned tree (branch `sd-clean`, rebuilt from stock
`upstream/main` with only the SparseDrive V1 geometry-only and V1.5 file sets)
is behaviour-preserving, by re-measuring each model on its home domain and
checking every cell against the number pinned before the refactor.

Verdict: **every re-measured cell reproduces.** Largest deviation across the
whole matrix is 0.005 absolute, on navhard stage-one.

References: nuScenes from `docs/declutter_experiment_plan.md` section 8;
NAVSIM planning from `work_dirs/navsim_eval/abc_matrix_metrics.csv` (column `A`)
and, for V1.5, the pre-refactor scoring runs in
`work_dirs/navsim_eval/epdms_navtest_v2/` and `epdms_v2/`.

## Panel 1 — nuScenes

Stock 10-class, 6-step, rescore-off, `withmap` infos, so detection, map, L2 and
collision are all directly comparable to the official checkpoint. One 8-GPU
Lilypad job, five evals in parallel (`evalclean_nuscenes_panel.yaml`).

| Model | det mAP | NDS | map mAP | L2 avg (m) | col avg (%) |
|---|---|---|---|---|---|
| official `sparsedrive_stage2.pth` | 0.4138 | 0.5243 | 0.5654 | 0.6101 | 0.194 |
| V1 geometry-only, seed0 | 0.4131 | 0.5190 | 0.5656 | 0.6397 | 0.182 |
| V1 geometry-only, seed1 | 0.4094 | 0.5187 | 0.5554 | 0.6408 | 0.184 |
| V1 geometry-only, seed2 | 0.4094 | 0.5221 | 0.5527 | 0.6611 | 0.261 |
| **V1 geometry-only, mean** | **0.4106** | — | **0.5579** | **0.6472 ± 0.0121** | **0.209 ± 0.045** |
| V6c image-feature control | 0.4102 | 0.5212 | 0.5566 | 0.6012 | 0.174 |

Against the pinned references:

| Cell | reproduced | reference | delta |
|---|---|---|---|
| official det / map / L2 / col | 0.4138 / 0.5654 / 0.6101 / 0.194 | 0.414 / 0.565 / 0.611 / 0.196 | -0.0002 / +0.0004 / -0.0009 / -0.002 |
| V1 mean det / map / L2 / col | 0.4106 / 0.5579 / 0.6472 / 0.209 | 0.410 / 0.558 / 0.647 ± 0.011 / 0.207 ± 0.038 | +0.0006 / -0.0001 / +0.0002 / +0.002 |
| V6c det / map / L2 / col | 0.4102 / 0.5566 / 0.6012 / 0.174 | 0.411 / 0.557 / 0.601 / 0.173 | -0.0008 / -0.0004 / +0.0002 / +0.001 |

The headline effect of the geometry-only planner reproduces exactly: V1 mean
minus V6c control is **+0.0460 m L2**, against the +0.046 recorded in the
declutter plan. Both arms share the trainer, data, schedule and seed protocol;
the only difference is whether the planner reads image features.

Note the det/map cells are not visible in wandb. `lilypad_entrypoint_nuscenes`
scrapes only the planning PrettyTable (`_parse_planning_table`), so detection
and map are recoverable only from the per-eval logs uploaded to
`s3://research-datasets-chicago/users/tejan/navsim/sparsedrive/eval_logs/sd_clean/nuscenes_panel/`,
where map mAP is printed as `mAP_normal`.

## Panel 2 — NAVSIM

Planning only. Inference ran on Lilypad (8 shards each); scoring ran locally,
see "cluster scoring is not available" below.

| Model | navtest PDMS | navtest-v2 EPDMS | navhard EPDMS (stage 1 / stage 2) |
|---|---|---|---|
| official SparseDriveV2 | 0.9222 | 0.9037 | 0.3753 shipped, 0.4046 fix-bug |
| V1 geometry-only (A) | 0.7629 | 0.7639 | 0.1929 (0.5305 / 0.3469) |
| V1.5 trajectory-vocabulary | 0.8613 | 0.8578 | 0.3281 (0.6862 / 0.4682) |

Against the pinned references:

| Cell | reproduced | reference | delta |
|---|---|---|---|
| A navtest PDMS | 0.7629 | 0.7620 | +0.0009 |
| A navtest-v2 EPDMS | 0.7639 | 0.7634 | +0.0005 |
| A navhard combined | 0.19287 | 0.1917 | +0.0012 |
| A navhard stage one / two | 0.53051 / 0.34691 | 0.5258 / 0.3464 | +0.0047 / +0.0005 |
| V1.5 navtest PDMS | 0.8613 | 0.8613 | 0.0000 |
| V1.5 navtest-v2 EPDMS | 0.8578 | 0.8581 | -0.0002 |
| V1.5 navhard combined | 0.32807 | 0.3297 | -0.0016 |
| V1.5 navhard stage one / two | 0.68622 / 0.46819 | 0.687 / 0.471 | -0.0008 / -0.0028 |

The SparseDriveV2 row is an external baseline carried forward unchanged, not a
reproduction: re-running it would test their code, not this refactor. Its
perception cells are N/A because it ships no detection or map head, and it
trained on 1,192 navtrain logs against our 1,067, so the planning gap is not a
like-for-like architecture comparison.

### Not reproduced on the clean tree

- **NAVSIM detection and map.** No such job was launched. Values carried
  forward for A: det mAP 0.3415, det NDS 0.3981, map mAP 0.8425.
- **The imgfeat control on NAVSIM.** Carried forward: navtest PDMS 0.8060,
  navtest-v2 EPDMS 0.8051, navhard combined 0.1977. The geometry-only cost on
  NAVSIM therefore mixes trees and should be read as indicative only; the
  nuScenes panel measures that same contrast entirely within the clean tree.
- **B-v1 and B-v2.** Dropped from the comparison because their map cells are
  confounded by a `ground_height` mismatch between training and evaluation.
  Configs and checkpoints are retained so the rows stay reproducible.

## Trajectories are not bit-identical

The clean tree's navhard trajectories differ from the pre-refactor pickles:
median per-token max deviation 1.0 cm, worst case 3.9 m, and zero of 5,912
tokens match exactly. This is hardware, not the refactor. The references were
produced locally on an RTX 5090; these ran on A100s, so kernel selection and
reduction order differ. Aggregate scores are the meaningful check, and they
agree to within 0.005 across every cell above.

A bit-exactness gate would need the same GPU for both arms.

## Cluster scoring is not available for navhard or navtest-v2

Neither column can be produced by a Lilypad job.

- **navhard.** Inference succeeds and exports trajectories, then scoring dies
  in `navsim.planning.utils.multithreading.worker_ray_no_torch.RayDistributedNoTorch`
  with `ValueError: When connecting to an existing cluster, num_cpus and
  num_gpus must not be provided`. The worker calls `ray.init()` with explicit
  resources, which Ray rejects because the Lilypad job is already a Ray
  cluster. This predates the refactor: the pinned navhard numbers were produced
  by `work_dirs/navsim_eval/run_v6_navhard_local.sh`, whose header records that
  cluster evals were already paused.
- **navtest-v2.** `lilypad_entrypoint_navsim` wires only `pdm_v1` and
  `epdms_two_stage`; there is no navtest-v2 mode.

Both are cheap to score locally from the exported trajectories, about 5 minutes
per model, so the split is: cluster does inference, local does scoring. See
`lilypad_config/navsim_eval/score_navhard_local.sh` and
`score_navtest_v2_local.sh`. The navtest-v2 metric cache lives with the
SparseDriveV2 harness as `exp/metric_cache_navtestv2`, not under
`SparseDrive/work_dirs` with the others.

## Reproducing

```bash
# nuScenes panel and NAVSIM inference (5 Lilypad jobs)
cd lilypad_config/navsim_eval && bash submit.sh evalclean_

# navhard and navtest-v2 scoring, after the jobs export trajectories
bash lilypad_config/navsim_eval/score_navhard_local.sh
bash lilypad_config/navsim_eval/score_navtest_v2_local.sh
```

Job specs pin `num_gpus` inside `entrypoint_fn_config`, not only under
`cluster_resources`. The eval driver reads it from the config block with a
default of 1, so omitting it makes any `num_shards > 1` fail validation before
staging.
