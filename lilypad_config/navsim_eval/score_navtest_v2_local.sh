#!/bin/bash
# Local navtest-v2 EPDMS scoring for the sd-clean tree.
#
# navtest-v2 is not a cluster mode: lilypad_entrypoint_navsim only wires
# pdm_v1 and epdms_two_stage. The cluster jobs supply the trajectories and
# this scores them, reusing the navtest-v2 metric cache that lives with the
# SparseDriveV2 harness rather than under SparseDrive/work_dirs.
set -u
CLEAN=/home/tejan/lwm-rl/SparseDrive
EV=/home/tejan/lwm-rl/SparseDrive/work_dirs/navsim_eval
CACHE=/home/tejan/lwm-rl/SparseDriveV2/exp/metric_cache_navtestv2
LP_PY=/home/tejan/miniconda3/envs/lilypad/bin/python
cd "$CLEAN"
mkdir -p "$EV/logs"

for tag in clean_a_iter8703_navtest clean_sd15_iter8703_navtest; do
  echo "SCORE_START $tag $(date +%H:%M:%S)"
  $LP_PY navsim_agent/score_epdms_navtest_v2.py \
      --agent "traj:$EV/trajs_${tag}.pkl" \
      --split navtest \
      --metric-cache "$CACHE" \
      --scorer fix_bug \
      --worker ray_distributed_no_torch \
      --run-tag "${tag}v2" \
      > "$EV/logs/epdms_${tag}v2_local.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "SCORE_OK $tag $(date +%H:%M:%S)"
  else
    echo "SCORE_FAIL $tag rc=$rc $(date +%H:%M:%S)"
    tail -20 "$EV/logs/epdms_${tag}v2_local.log"
  fi
done
echo "SCORE_ALL_DONE"
