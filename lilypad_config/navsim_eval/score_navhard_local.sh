#!/bin/bash
# Local navhard two-stage EPDMS scoring for the sd-clean tree.
#
# The cluster jobs complete inference and export trajectories, then fail in
# scoring: navsim's RayDistributedNoTorch calls ray.init() with explicit
# num_cpus/num_gpus, which Ray rejects inside the Lilypad job's existing
# cluster. The reference numbers in abc_matrix_metrics.csv were produced the
# same way, by scoring locally (work_dirs/navsim_eval/run_v6_navhard_local.sh).
#
# Inference is already done on the cluster, so this only re-scores the
# downloaded trajectory pickles.
set -u
CLEAN=/home/tejan/lwm-rl/.wtc/worktrees/sd-clean/SparseDrive
EV=/home/tejan/lwm-rl/SparseDrive/work_dirs/navsim_eval
LP_PY=/home/tejan/miniconda3/envs/lilypad/bin/python
cd "$CLEAN"
mkdir -p "$EV/logs"

for tag in clean_a_iter8703_navhard2s clean_sd15_iter8703_navhard2s; do
  echo "SCORE_START $tag $(date +%H:%M:%S)"
  $LP_PY navsim_agent/score_epdms_two_stage.py \
      --agent "traj:$EV/trajs_${tag}.pkl" \
      --split navhard_two_stage \
      --metric-cache "$EV/metric_cache_navhard2s_full" \
      --worker ray_distributed_no_torch \
      --run-tag "$tag" \
      > "$EV/logs/epdms_${tag}_local.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "SCORE_OK $tag $(date +%H:%M:%S)"
    grep -aiE "^(combined|stage_one|stage_two)|epdms" "$EV/logs/epdms_${tag}_local.log" | tail -8
  else
    echo "SCORE_FAIL $tag rc=$rc $(date +%H:%M:%S)"
    tail -20 "$EV/logs/epdms_${tag}_local.log"
  fi
done
echo "SCORE_ALL_DONE"
