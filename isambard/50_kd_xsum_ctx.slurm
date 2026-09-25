#!/bin/bash
#SBATCH --job-name=xsum-ctxkd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/ctxkd-%j.out
#SBATCH --error=logs/ctxkd-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# The CONTEXT-DISTILLATION arm, run alongside the shared-prompt one.
#
# Every other arm gives the teacher and the student the same prompt, so the
# teacher's only advantage is its weights - each half-expert saw data the merge
# only half-absorbed. Here the teacher is additionally conditioned on a strict
# system prompt (20-25 words, present tense, third person, no preamble) while
# the student sees "Summarise this BBC News article in one sentence:". The
# student therefore has to absorb the format into its weights rather than read
# it each time.
#
# THIS IS NOT COMPARABLE WITH KD.md. Two variables move at once relative to the
# record - the alignment fix and the prompt - so the arm carries its own dataset
# tree and re-measures its own baselines under the student prompt. Read it
# against the baselines it produces, never against 0.3987/0.3970/0.3990/0.4049.
#
# Runs in parallel with 30_kd_xsum.slurm: separate datasets, separate outputs.
# The one shared artefact is the merge, which it reads and never writes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

MERGE="$MODELS_DIR/gemma3-xsum-merged-linear"
STUDENT="$MODELS_DIR/gemma3-xsum-ctxkd"

echo "=== $(hostname) | $(date -Is) ==="
[ -d "$MERGE" ] || { echo "merge missing at $MERGE - run 30_kd_xsum.slurm first" >&2; exit 1; }

# gemma-3-4b-it is multimodal, so axolotl loads an image processor alongside the
# tokenizer. mergekit writes weights and tokenizer files but not
# preprocessor_config.json, and training dies with "Can't load image processor"
# AFTER the teacher pass - 33 minutes in. stage_model_dir.py symlinks the merge
# and fills only the missing JSON from the base model.
if [ ! -f "$MERGE/preprocessor_config.json" ]; then
  echo "=== staging processor files into the merge ==="
  mv "$MERGE" "$MERGE.raw"
  uv run --project evaluate python stage_model_dir.py \
    --source "$MERGE.raw" --dest "$MERGE" --base-model google/gemma-3-4b-it
fi

# --- 1. the ctx dataset tree -----------------------------------------------
if [ ! -d "$DATASETS_DIR/xsum_ctx_dataset" ]; then
  echo "=== building the ctx datasets ==="
  XSUM_VARIANT=ctx uv run --project evaluate python prepare_xsum_data.py
fi

# --- 2. teacher logprobs, strict prompt, each expert over its own half ------
for i in 1 2; do
  OUT="$DATASETS_DIR/xsum_ctxkd_half$i"
  [ -f "$OUT/.kd_alignment_v2" ] && { echo "=== teacher $i present ==="; continue; }
  rm -rf "$OUT"
  echo "=== teacher $i logprobs (strict prompt) ==="
  uv run --project evaluate python precompute_logprobs_ctx.py \
    --model google/gemma-3-4b-it \
    --adapter "$MODELS_DIR/gemma3-xsum-$i-of-2-lora" \
    --dataset "$DATASETS_DIR/xsum_ctx_dataset$i/train" \
    --output "$OUT" --verify 32
done

# Marker, not directory existence. The run that crashed in
# concat_datasets.py wrote the dataset and died before stamping it, so a
# -d test skipped the rebuild and left an unmarked directory that the
# sweep arms then refused.
if [ ! -f "$DATASETS_DIR/xsum_ctxkd_train/.kd_alignment_v2" ]; then
  rm -rf "$DATASETS_DIR/xsum_ctxkd_train"
  echo "=== combining ==="
  uv run --project evaluate python concat_datasets.py \
    "$DATASETS_DIR/xsum_ctxkd_half1" "$DATASETS_DIR/xsum_ctxkd_half2" \
    --output "$DATASETS_DIR/xsum_ctxkd_train"
fi

# --- 3. distil --------------------------------------------------------------
echo "=== training (0.2 / 1.0, strict teacher / terse student) ==="
# axolotl writes a LoRA adapter, so a rerun that retrains is pure waste - and
# these jobs have already been resubmitted twice for failures downstream of
# training.
if [ -f "$STUDENT/adapter_model.safetensors" ]; then
  echo "=== student already trained at $STUDENT, skipping ==="
else
  cd "$REPO_ROOT/train"
  uv run axolotl train xsum_gemma_ctxkd.yaml
  cd "$REPO_ROOT"
fi

# --- 4. evaluate, all of it on the STUDENT prompt ---------------------------
# The baselines are re-measured here rather than carried over. A model asked
# with the terse prompt is not the same measurement as the same model asked with
# the instruction-heavy one, and the whole point of the arm is that the prompt
# moved.
echo "=== evaluating on the ctx test set ==="
cd "$REPO_ROOT/evaluate"
evaluate () {
  local name=$1 model=$2 adapter=${3:-}
  local args=(--model "$model" --dataset "$DATASETS_DIR/xsum_ctx_dataset/test"
              --output "$RESULTS_DIR/$name.json")
  [ -n "$adapter" ] && args+=(--adapter "$adapter")
  uv run python evaluate_summarisation.py "${args[@]}"
}

# The student is an adapter on the merge - see 30_kd_xsum.slurm.
evaluate ctx-kd     "$MERGE" "$STUDENT"
evaluate ctx-merge  "$MERGE"
evaluate ctx-half1  google/gemma-3-4b-it "$MODELS_DIR/gemma3-xsum-1-of-2-lora"
evaluate ctx-half2  google/gemma-3-4b-it "$MODELS_DIR/gemma3-xsum-2-of-2-lora"
evaluate ctx-full   google/gemma-3-4b-it "$MODELS_DIR/gemma3-xsum-full-lora"

echo "=== done. Results in $RESULTS_DIR (ctx-*.json) ==="
