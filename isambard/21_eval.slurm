#!/bin/bash
#SBATCH --job-name=xsum-eval
#SBATCH --array=1-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-%A_%a.out
#SBATCH --error=logs/eval-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# One evaluation per array task. Fourteen of them run in the wall-clock time of
# the slowest, where the previous single job ran them back to back inside one
# three-hour allocation; the GPU-hours are the same, the critical path is not.
#
#   1      the 8-way merge          1/1 of the data, reached by averaging
#   2-5    the four pair merges     1/4 each
#   6-13   the eight shard experts  1/8 each
#   14     the full-data expert     the reference every rung is read against
#
# Submit after the merges succeed:
#   sbatch --dependency=afterok:<merge-array-id> isambard/21_eval.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

SPLITS="${SPLITS:-8}"
BASE="google/gemma-3-4b-it"
i="$SLURM_ARRAY_TASK_ID"

# name|model|adapter  - adapter empty for a fully merged model.
SPECS=()
SPECS+=("merge-${SPLITS}way|$MODELS_DIR/gemma3-xsum-merged-linear-${SPLITS}way|")
# Pair merges exist only in the 8-shard design. Including them unconditionally
# made array indices 2-5 of a quarter run point at models that were never built.
for a in $( [ "$SPLITS" = "8" ] && echo 1 3 5 7 ); do
  b=$(( a + 1 ))
  SPECS+=("pair${a}${b}|$MODELS_DIR/gemma3-xsum-merged-pair${a}${b}|")
done
for s in $(seq 1 "$SPLITS"); do
  SPECS+=("shard${s}-of-${SPLITS}|$BASE|$MODELS_DIR/gemma3-xsum-${s}-of-${SPLITS}-lora")
done
SPECS+=("full|$BASE|$MODELS_DIR/gemma3-xsum-full-lora")

if [ "$i" -lt 1 ] || [ "$i" -gt "${#SPECS[@]}" ]; then
  echo "array index $i outside 1..${#SPECS[@]}" >&2; exit 1
fi
IFS='|' read -r NAME MODEL ADAPTER <<<"${SPECS[$((i - 1))]}"

echo "=== eval $NAME | $(hostname) | $(date -Is) ==="
# Newer than the model it describes, not merely present. A result from a
# previous generation of an adapter is the most dangerous kind of stale
# artefact: it looks exactly like a measurement.
TARGET="${ADAPTER:-$MODEL}"
if [ -f "$RESULTS_DIR/$NAME.json" ] \
   && { [ ! -e "$TARGET" ] || [ "$RESULTS_DIR/$NAME.json" -nt "$TARGET" ]; }; then
  echo "already evaluated against this model, skipping"; exit 0
fi

cd "$REPO_ROOT/evaluate"
ARGS=(--model "$MODEL" --dataset "$DATASETS_DIR/xsum_dataset/test"
      --output "$RESULTS_DIR/$NAME.json")
[ -n "$ADAPTER" ] && ARGS+=(--adapter "$ADAPTER")
uv run python evaluate_summarisation.py "${ARGS[@]}"

echo "=== $NAME done $(date -Is) ==="
