#!/bin/bash
#SBATCH --job-name=crosstask-merge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/crosstask-%j.out
#SBATCH --error=logs/crosstask-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# The comparator the routing arm lacks: one merged model doing both tasks,
# evaluated on exactly the examples routing was measured on.
#
# Routing scored 0.9850 crime and 0.4183 ROUGE-1 on the first 200 of each test
# set, with a perfect router - so that is the oracle ceiling. The cost of
# merging is the distance below it, and it has to be measured here rather than
# read off the Azure cross-task sweep, whose experts were a different
# generation on a different platform.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

MERGE="$MODELS_DIR/gemma3-crosstask-taskarith"
echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || echo "  (nvidia-smi unavailable)"

if [ ! -f "$MERGE/config.json" ]; then
  rm -rf "$MERGE" "$MERGE.raw"
  BASE_DIR=$(ls -d "$HF_HUB_CACHE"/models--google--gemma-3-4b-it/snapshots/*/ | head -1)
  echo "=== base resolved to $BASE_DIR ==="
  sed "s|LOCAL_BASE|${BASE_DIR%/}|" "$REPO_ROOT/merge/merge_crosstask_gemma_config.yaml" \
    > "$TMPDIR/crosstask-$SLURM_JOB_ID.yaml"
  cd "$REPO_ROOT/merge"
  uv run mergekit-yaml "$TMPDIR/crosstask-$SLURM_JOB_ID.yaml" "$MERGE" \
    --cuda --lazy-unpickle --lora-merge-cache "$TMPDIR/lora-$SLURM_JOB_ID"
  cd "$REPO_ROOT"
fi
if [ ! -f "$MERGE/preprocessor_config.json" ]; then
  mv "$MERGE" "$MERGE.raw"
  uv run --project evaluate python stage_model_dir.py \
    --source "$MERGE.raw" --dest "$MERGE" --base-model google/gemma-3-4b-it
fi

cd "$REPO_ROOT/evaluate"
echo "=== merged model on crime, same 200 as routing ==="
uv run python evaluate.py --model "$MERGE" \
  --dataset "$DATASETS_DIR/crime_dataset/test" --limit 200 \
  --output "$RESULTS_DIR/crosstask-merge-crime.json" 2>&1 | grep -E "^Accuracy|^Unparsed"
echo "=== merged model on xsum, same 200 as routing ==="
uv run python evaluate_summarisation.py --model "$MERGE" \
  --dataset "$DATASETS_DIR/xsum_dataset/test" --limit 200 \
  --output "$RESULTS_DIR/crosstask-merge-xsum.json" 2>&1 | grep -E "^rouge1|Length"
echo "=== done $(date -Is) ==="
