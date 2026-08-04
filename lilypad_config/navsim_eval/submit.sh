#!/bin/bash
# Submit NAVSIM SparseDrive jobs to Lilypad (same pattern as declutter/submit.sh).
# Usage: bash submit.sh stage1_overfit
set -e

PATTERN="${1:?usage: submit.sh <yaml-prefix, e.g. stage1_overfit>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LILYPAD="/home/tejan/miniconda3/envs/lilypad/bin/lilypad"

export AWS_ACCESS_KEY_ID=$(aws --profile oci.chi configure get aws_access_key_id)
export AWS_SECRET_ACCESS_KEY=$(aws --profile oci.chi configure get aws_secret_access_key)
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('appliedintuition.wandb.io')[2])")

if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$WANDB_API_KEY" ]]; then
  echo "missing AWS or WANDB credentials" >&2
  exit 1
fi

IDS_FILE="$HERE/submitted_${PATTERN}_$(date +%Y%m%d_%H%M%S).txt"
for YAML in "$HERE"/${PATTERN}*.yaml; do
  [[ -e "$YAML" ]] || { echo "no yamls match ${PATTERN}*"; exit 1; }
  NAME=$(basename "$YAML" .yaml)
  # NAVSIM evals spend long stretches in CPU staging/scoring or low-SM
  # geometry-only inference.  Opt out so the idle-GPU reaper cannot discard
  # a healthy full evaluation before its first shard completes.
  OUTPUT=$("$LILYPAD" workload launch --ignore-idle-reaper "$YAML" 2>&1)
  WID=$(echo "$OUTPUT" | grep -oP '(?<=Workload launched with ID: )\S+' || true)
  if [[ -z "$WID" ]]; then
    echo "FAILED to launch $NAME:"
    echo "$OUTPUT"
    exit 1
  fi
  echo "$NAME  $WID" | tee -a "$IDS_FILE"
done
echo ""
echo "Workload IDs saved to $IDS_FILE"
