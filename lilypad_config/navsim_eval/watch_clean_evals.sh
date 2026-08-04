#!/bin/bash
# Poll the sd-clean eval workloads and print a status line per poll.
# Prints RESULT lines the parent agent watches for terminal states.
export AWS_PROFILE=oci.chi
export AWS_ACCESS_KEY_ID=$(aws --profile oci.chi configure get aws_access_key_id)
export AWS_SECRET_ACCESS_KEY=$(aws --profile oci.chi configure get aws_secret_access_key)
LP=/home/tejan/miniconda3/envs/lilypad/bin/lilypad
WORKLOADS="sd_evalclean_navtest_a-q4ttkp sd_evalclean_navtest_sd15-v4d7a2 sd_evalclean_navhard_a-3k99x2 sd_evalclean_navhard_sd15-990p14 sd_evalclean_nuscenes_panel-49fu6n"
N_WORKLOADS=5

while true; do
  done_count=0
  for w in $WORKLOADS; do
    info=$($LP workload info "$w" 2>/dev/null | grep -viE "circuit_breaker|^INFO")
    st=$(echo "$info" | awk '/^Status/{print $2}')
    dur=$(echo "$info" | awk '/^Duration/{print $2}')
    echo "$(date +%H:%M:%S) ${w%%-*} $st $dur"
    case "$st" in
      *COMPLETED*|*SUCCEEDED*|*FAILED*|*STOPPED*|*CANCELLED*|*ERROR*)
        echo "RESULT_TERMINAL $w $st"
        done_count=$((done_count+1)) ;;
    esac
  done
  if [ "$done_count" -eq "$N_WORKLOADS" ]; then echo "RESULT_ALL_TERMINAL"; break; fi
  echo "---"
  sleep 180
done
