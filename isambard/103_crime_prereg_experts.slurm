#!/bin/bash
#SBATCH --job-name=crime-prereg-x
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --array=1-16
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Pre-registered replication, seeds 9-16. See PREREGISTRATION-twoway.md, which
# was committed before any of these experts existed: n is fixed at eight, the
# seeds do not overlap the earlier arm, and no seed will be added whatever the
# interval does.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
i="$SLURM_ARRAY_TASK_ID"
HALF=$(( (i - 1) / 8 + 1 ))
SEED=$(( (i - 1) % 8 + 9 ))
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
