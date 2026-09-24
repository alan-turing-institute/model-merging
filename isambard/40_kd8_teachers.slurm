#!/bin/bash
#SBATCH --job-name=xsum-kd8-teachers
#SBATCH --array=1-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/kd8-teacher-%A_%a.out
#SBATCH --error=logs/kd8-teacher-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Teacher logprobs for the 8-shard KD arm: shard expert i over shard i, one per
# array task. Each expert supplies logprobs only for the data it was trained
# on, which keeps the procedure inside the "poor man's parallelism" framing -
# nothing here needs a machine that has seen all the data.
#
# These are produced ONCE and consumed by all three arms of the alpha sweep.
# Recomputing them per arm would triple the most expensive step in the chain
# for three runs that are supposed to differ only in two scalars.
#
#   sbatch --dependency=afterok:<shard-array-id> isambard/40_kd8_teachers.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

SPLITS="${SPLITS:-8}"
i="$SLURM_ARRAY_TASK_ID"
OUT="$DATASETS_DIR/xsum_kd${SPLITS}_shard${i}"

echo "=== teacher $i of $SPLITS | $(hostname) | $(date -Is) ==="

# --- the alignment gate ----------------------------------------------------
# Seconds, and the whole reason this arm is being re-run. A teacher==student KL
# that is not ~0 means the producer is still misplacing rows, and everything
# downstream trains on noise - silently, which is how the last month of KD
# numbers were generated (KD.md, 2026-09-24). Per-task output file: eight tasks
# writing one path would race.
CHECK="$RESULTS_DIR/alignment_check_shard${i}.txt"
uv run --project evaluate python experiments/kd_alignment_smoke.py --examples 4 \
  | tee "$CHECK"
grep -qE 'one extra row.*0\.0000' "$CHECK" \
  || { echo "ALIGNMENT CHECK FAILED - refusing to produce teacher data" >&2; exit 1; }

# --- teacher pass ----------------------------------------------------------
# precompute_logprobs.py writes .kd_alignment_v2 itself; its presence is what
# lets a resubmitted array skip a shard that is already correct, and what makes
# a pre-fix dataset refuse to be consumed rather than silently train on it.
if [ -f "$OUT/.kd_alignment_v2" ]; then
  echo "=== shard $i already aligned at $OUT, skipping ==="
  exit 0
fi
rm -rf "$OUT"

uv run --project evaluate python precompute_logprobs.py \
  --model google/gemma-3-4b-it \
  --adapter "$MODELS_DIR/gemma3-xsum-${i}-of-${SPLITS}-lora" \
  --dataset "$DATASETS_DIR/xsum_dataset${i}/train" \
  --output "$OUT" --verify 32
touch "$OUT/.precompute_complete"

echo "=== teacher $i done $(date -Is) ==="
