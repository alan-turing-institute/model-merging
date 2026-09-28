#!/bin/bash
#SBATCH --job-name=method-sweep
#SBATCH --array=1-14%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:30:00
#SBATCH --output=logs/method-sweep-%A_%a.out
#SBATCH --error=logs/method-sweep-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Seven mergekit methods on both tasks, current generation.
#
# WHY. The method sweep on record ran on the retired Azure generation, and it
# holds the largest effect in the programme: TIES recovered R = +0.49 where
# linear scored -0.24, a swing of 0.73 - bigger than every distillation and MoE
# effect combined. Two things are wrong with citing it as it stands. Its
# recovery fractions are computed against that generation's baselines, so they
# are not comparable with anything measured on Isambard. And XSum was never
# swept at all: the headline summarisation null (merge 0.3990 against a better
# half of 0.3977) is a statement about LINEAR merging only. If TIES gains there
# what it gained on crime, that null is method-specific rather than a property
# of merging.
#
# SERIALISED ON PURPOSE (%1). Each cell materialises a full ~8 GB model, and
# the home quota is 101 GB with roughly 27 GB free. Two concurrent cells plus
# the re-downloaded base would exceed it, and a full disk in this project has
# presented as three unrelated-looking application errors. One at a time, each
# deleted before the next starts.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

METHODS=(linear ties dare_ties slerp task_arithmetic model_stock arcee_fusion)
i="$SLURM_ARRAY_TASK_ID"
if [ "$i" -le 7 ]; then
  TASK=crime; METHOD="${METHODS[$((i - 1))]}"
else
  TASK=xsum;  METHOD="${METHODS[$((i - 8))]}"
fi

NAME="sweep-${TASK}-${METHOD}"
OUT="$MODELS_DIR/$NAME"
CONFIG="merge_sweep_${TASK}_${METHOD}_config.yaml"
RESULT="$RESULTS_DIR/${NAME}.json"

echo "=== $NAME | $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || echo '  (nvidia-smi unavailable on this node)'

if [ -f "$RESULT" ]; then
  echo "=== already evaluated, skipping ==="; exit 0
fi

[ -f "$REPO_ROOT/merge/$CONFIG" ] || { echo "no config at merge/$CONFIG" >&2; exit 1; }
for n in 1 2; do
  [ -f "$MODELS_DIR/gemma3-${TASK}-${n}-of-2-lora/adapter_model.safetensors" ] \
    || { echo "expert $n missing for $TASK" >&2; exit 1; }
done
grep -E '^merge_method:' "$REPO_ROOT/merge/$CONFIG"
grep -E 'gemma3-' "$REPO_ROOT/merge/$CONFIG"

# Both the model and the LoRA cache go away on ANY exit. A cell that dies inside
# mergekit otherwise leaves ~8 GB behind, and fourteen of those is the quota.
LORA_CACHE="$TMPDIR/lora-cache-$SLURM_JOB_ID-$i"
cleanup() { rm -rf "$LORA_CACHE" "$OUT" "$OUT.raw"; }
trap cleanup EXIT

rm -rf "$OUT" "$OUT.raw"
echo "=== merging ($METHOD) ==="
cd "$REPO_ROOT/merge"
uv run mergekit-yaml "$CONFIG" "$OUT" \
  --cuda --lazy-unpickle --lora-merge-cache "$LORA_CACHE"
cd "$REPO_ROOT"

# A real artefact, not a directory: a run that dies inside mergekit leaves an
# empty $OUT behind, and a -d test would then sail past it.
[ -f "$OUT/config.json" ] || { echo "merge produced no config.json" >&2; exit 1; }

# gemma-3-4b-it is multimodal; mergekit writes weights and tokenizer files but
# not preprocessor_config.json, and evaluation dies on "Can't load image
# processor" after the model is already in memory.
if [ ! -f "$OUT/preprocessor_config.json" ]; then
  echo "=== staging processor files ==="
  mv "$OUT" "$OUT.raw"
  uv run --project evaluate python stage_model_dir.py \
    --source "$OUT.raw" --dest "$OUT" --base-model google/gemma-3-4b-it
fi

echo "=== evaluating ==="
cd "$REPO_ROOT/evaluate"
if [ "$TASK" = crime ]; then
  uv run python evaluate.py \
    --model "$OUT" --dataset "$DATASETS_DIR/crime_dataset/test" --output "$RESULT"
else
  uv run python evaluate_summarisation.py \
    --model "$OUT" --dataset "$DATASETS_DIR/xsum_dataset/test" --output "$RESULT"
fi

[ -f "$RESULT" ] || { echo "evaluation wrote no result" >&2; exit 1; }
echo "=== $NAME done $(date -Is) ==="
