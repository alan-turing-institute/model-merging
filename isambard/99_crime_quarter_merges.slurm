#!/bin/bash
#SBATCH --job-name=crime-q-merge
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=1-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Linear merge of the four crime quarter-experts, one per expert seed, scored on
# the same 1,077 posts as the experts themselves.
#
# The merged model is deleted once it has been scored: it is 8.1 GB and fully
# reproducible from its four adapters and the config, so keeping five of them
# costs 40 GB of project storage to save a few minutes of re-merging.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
SEED="$SLURM_ARRAY_TASK_ID"
CONFIG="merge_linear_crime_4way_seed${SEED}_config.yaml"
OUT="$MODELS_DIR/crime-4way-seed${SEED}"
RESULT="$RESULTS_DIR/crime-4way-merge-seed${SEED}.json"

echo "=== 4-way linear merge, expert seed $SEED | $(hostname) | $(date -Is) ==="
for q in 1 2 3 4; do
  [ -f "$MODELS_DIR/gemma3-crime-q${q}-seed${SEED}-lora/adapter_model.safetensors" ] \
    || { echo "quarter $q expert for seed $SEED missing" >&2; exit 1; }
done
grep -E '^merge_method:' "$REPO_ROOT/merge/$CONFIG"
grep 'gemma3-crime' "$REPO_ROOT/merge/$CONFIG"

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
  uv run --project evaluate python stage_model_dir.py \
    --source "$OUT.raw" --dest "$OUT" --base-model google/gemma-3-4b-it
fi
cd "$REPO_ROOT/evaluate"
uv run python evaluate.py --model "$OUT" \
  --dataset "$DATASETS_DIR/crime_dataset/test" --output "$RESULT"
[ -f "$RESULT" ] || { echo "no result written" >&2; exit 1; }
echo "=== seed $SEED done $(date -Is) ==="
