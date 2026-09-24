#!/bin/bash
#SBATCH --job-name=xsum-merge-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/merge-eval-%j.out
#SBATCH --error=logs/merge-eval-%j.out
##SBATCH --account=CHANGEME
##SBATCH --partition=workq
#
# Submit with a dependency so it starts only if every shard trained:
#   sbatch --dependency=afterok:<array-job-id> isambard/20_merge_eval.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

SPLITS="${SPLITS:-8}"
MERGED="$MODELS_DIR/gemma3-xsum-merged-linear-${SPLITS}way"

echo "=== merging $SPLITS shard experts $(date -Is) ==="
cd "$REPO_ROOT/merge"
uv run mergekit-yaml "merge_linear_xsum_${SPLITS}way_config.yaml" "$MERGED" \
  --cuda --lazy-unpickle --lora-merge-cache "$REPO_ROOT/.lora-cache"

echo "=== evaluating $(date -Is) ==="
cd "$REPO_ROOT/evaluate"
evaluate () {   # name, model path, adapter (may be empty)
  local name=$1 model=$2 adapter=${3:-}
  local args=(--model "$model" --dataset "$DATASETS_DIR/xsum_dataset/test"
              --output "$RESULTS_DIR/$name.json")
  [ -n "$adapter" ] && args+=(--adapter "$adapter")
  uv run python evaluate_summarisation.py "${args[@]}"
}

evaluate "merge-${SPLITS}way" "$MERGED"
for i in $(seq 1 "$SPLITS"); do
  evaluate "shard${i}-of-${SPLITS}" google/gemma-3-4b-it \
    "$MODELS_DIR/gemma3-xsum-${i}-of-${SPLITS}-lora"
done
evaluate "full" google/gemma-3-4b-it "$MODELS_DIR/gemma3-xsum-full-lora"

echo "=== done. Results in $RESULTS_DIR ==="
echo "Next: uv run --project evaluate python experiments/bootstrap_rouge.py $RESULTS_DIR"
