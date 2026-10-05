#!/bin/bash
#SBATCH --job-name=crime-q-experts
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --array=1-20
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Four-way crime shard experts: four quarters x five expert seeds.
#
# WHY. The two-way arm cannot answer the question the report actually asks of
# poor man's parallelism. If data cannot be centralised, the alternative to
# merging is simply to keep the better shard, so the comparison that decides
# feasibility is merge against BEST SHARD - which is what the recovery fraction
# measures. At two-way that gap is 0.0037, about one expert-draw standard
# deviation, from a single pair: not a result.
#
# The BillSum scaling curve showed the cause is the denominator, not the method:
# full-data training beats one shard by more as the shards get smaller. Four-way
# is the smallest split that buys a denominator worth dividing by while leaving
# each expert 2,153 examples to train on.
#
# Each task trains one expert and scores it on the full 1,077-post test set, so
# the per-seed best shard is measured rather than assumed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
i="$SLURM_ARRAY_TASK_ID"
Q=$(( (i - 1) / 5 + 1 ))
SEED=$(( (i - 1) % 5 + 1 ))
CONFIG="crime_gemma_q${Q}_seed${SEED}.yaml"
OUT="$MODELS_DIR/gemma3-crime-q${Q}-seed${SEED}-lora"
RESULT="$RESULTS_DIR/crime-q${Q}-seed${SEED}.json"

echo "=== crime quarter $Q, expert seed $SEED | $(hostname) | $(date -Is) ==="
grep -E '^seed:|^output_dir:' "$REPO_ROOT/train/$CONFIG"
nvidia-smi --query-gpu=name --format=csv,noheader || true

if [ ! -f "$OUT/adapter_model.safetensors" ]; then
  cd "$REPO_ROOT/train"
  uv run axolotl train "$CONFIG"
  [ -f "$OUT/adapter_model.safetensors" ] || { echo "no adapter written" >&2; exit 1; }
  cd "$REPO_ROOT"
else
  echo "=== adapter already present, skipping training ==="
fi

if [ ! -f "$RESULT" ]; then
  cd "$REPO_ROOT/evaluate"
  uv run python evaluate.py --model google/gemma-3-4b-it --adapter "$OUT" \
    --dataset "$DATASETS_DIR/crime_dataset/test" --output "$RESULT"
  [ -f "$RESULT" ] || { echo "no result written" >&2; exit 1; }
fi
echo "=== done $(date -Is) ==="
