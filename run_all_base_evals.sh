#!/bin/bash
# Runs the full CORE benchmark (scripts/base_eval.py) over every checkpoint
# of every run listed below, one run directory at a time. Each run directory
# becomes its own wandb run (named after the directory), sweeping every
# checkpoint found in it (--steps all). Continues past failures: a failure
# in one run/checkpoint is logged and skipped, the rest still run.
#
# Usage: ./run_all_base_evals.sh

set -uo pipefail # NOT set -e: we want to keep going if one run fails

PARENT_DIR="/root/models"
RUN_DIRS=(
  "d20-edu"
  "d20-random"
  "d32-hq"
  "d32-our"
  "run-1-fineweb2-pdf-filtered"
  "run-2-fineweb2-rehyd-pdf"
  "run-3-multinode-test"
  "run-5-rehyd-removal-rate"
  "run-6-rehydrated-removal-rate-filtered"
  "run-7-filtered-ge3-removal-rate"
  "run-8-fineweb-hq-removal-rate"
  "run-9-multinode-fineweb2-finepdfs-filtered-ge25"
  "run-10-multinode-finewebhq"
)

export WANDB_ENTITY="fineweb2edupl"
export NANOCHAT_BASE_DIR="/root/FinetextPL-Edu/nanochat-pl"

cd "$NANOCHAT_BASE_DIR"

FAILED_RUNS=()

for run_name in "${RUN_DIRS[@]}"; do
  checkpoint_dir="$PARENT_DIR/$run_name"

  if [ ! -d "$checkpoint_dir" ]; then
    echo "=== SKIPPING $run_name: directory not found at $checkpoint_dir ==="
    FAILED_RUNS+=("$run_name (directory not found)")
    continue
  fi

  echo "=== Evaluating $run_name ($checkpoint_dir) ==="
  python -m scripts.base_eval \
    --checkpoint-dir "$checkpoint_dir" \
    --steps all \
    --run "$run_name"
  status=$?

  if [ $status -ne 0 ]; then
    echo "=== FAILED: $run_name (exit code $status) ==="
    FAILED_RUNS+=("$run_name (exit code $status)")
  else
    echo "=== Done: $run_name ==="
  fi
done

echo
echo "================ Summary ================"
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
  echo "All ${#RUN_DIRS[@]} runs completed successfully."
else
  echo "${#FAILED_RUNS[@]} / ${#RUN_DIRS[@]} runs failed:"
  for f in "${FAILED_RUNS[@]}"; do
    echo "  - $f"
  done
fi
