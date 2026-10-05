#!/bin/bash
#SBATCH --job-name=crime-half-x
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --array=1-6
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Three more expert seeds for the two-way crime arm, trained and scored.
#
# WHY. Over the first five seeds the merge beat the better of its own two halves
# on every one, by +0.0059 on average - but the 95% interval was [-0.0002,
# +0.0121], missing zero by two ten-thousandths. The decision rule was fixed
# before that run and it was not met, so the result stands as not demonstrated.
# At the observed spread (sd 0.0049) eight seeds put the lower bound clear of
# zero if the effect is real, and leave it spanning zero if it is not. This is
# the cheapest way to settle a question that currently turns on rounding.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
i="$SLURM_ARRAY_TASK_ID"
HALF=$(( (i - 1) / 3 + 1 ))
SEED=$(( (i - 1) % 3 + 6 ))
CONFIG="crime_gemma${HALF}_seed${SEED}.yaml"
OUT="$MODELS_DIR/gemma3-crime-${HALF}-of-2-seed${SEED}-lora"
RESULT="$RESULTS_DIR/crime-${HALF}-of-2-seed${SEED}.json"
echo "=== half $HALF, seed $SEED | $(hostname) | $(date -Is) ==="
if [ ! -f "$OUT/adapter_model.safetensors" ]; then
  cd "$REPO_ROOT/train" && uv run axolotl train "$CONFIG"
  [ -f "$OUT/adapter_model.safetensors" ] || { echo "no adapter written" >&2; exit 1; }
  cd "$REPO_ROOT"
fi
if [ ! -f "$RESULT" ]; then
  cd "$REPO_ROOT/evaluate"
  uv run python evaluate.py --model google/gemma-3-4b-it --adapter "$OUT" \
    --dataset "$DATASETS_DIR/crime_dataset/test" --output "$RESULT"
fi
echo "=== done $(date -Is) ==="
