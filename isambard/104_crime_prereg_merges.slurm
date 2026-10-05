#!/bin/bash
#SBATCH --job-name=crime-prereg-m
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=1-8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Linear merges for the pre-registered replication, seeds 9-16.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
SEED=$(( SLURM_ARRAY_TASK_ID + 8 ))
NAME="seed${SEED}-crime-linear"
CONFIG="merge_seed${SEED}_crime_linear_config.yaml"
OUT="$MODELS_DIR/$NAME"; RESULT="$RESULTS_DIR/${NAME}.json"
echo "=== $NAME | $(hostname) | $(date -Is) ==="
for h in 1 2; do
  [ -f "$MODELS_DIR/gemma3-crime-${h}-of-2-seed${SEED}-lora/adapter_model.safetensors" ] \
    || { echo "half $h seed $SEED missing" >&2; exit 1; }
done
LORA_CACHE="$TMPDIR/lora-cache-$SLURM_JOB_ID-$SEED"
cleanup() { rm -rf "$LORA_CACHE" "$OUT" "$OUT.raw"; }
trap cleanup EXIT
rm -rf "$OUT" "$OUT.raw"
cd "$REPO_ROOT/merge"
uv run mergekit-yaml "$CONFIG" "$OUT" --cuda --lazy-unpickle --lora-merge-cache "$LORA_CACHE"
cd "$REPO_ROOT"
[ -f "$OUT/config.json" ] || { echo "merge produced no config.json" >&2; exit 1; }
if [ ! -f "$OUT/preprocessor_config.json" ]; then
  mv "$OUT" "$OUT.raw"
  uv run --project evaluate python stage_model_dir.py --source "$OUT.raw" --dest "$OUT" \
    --base-model google/gemma-3-4b-it
fi
cd "$REPO_ROOT/evaluate"
uv run python evaluate.py --model "$OUT" --dataset "$DATASETS_DIR/crime_dataset/test" --output "$RESULT"
echo "=== done $(date -Is) ==="
