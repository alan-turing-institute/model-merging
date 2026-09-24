#!/bin/bash
#SBATCH --job-name=xsum-kd-aligned
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/kd-xsum-%j.out
#SBATCH --error=logs/kd-xsum-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# The XSum KD arm re-run with the corrected teacher alignment (6f923a1).
#
# This does NOT use run_kd_pipeline.sh. That script resolves every artefact
# through the Azure ML registry and writes results to ~/cloudfiles; on Isambard
# there is no registry and every artefact is a path, so the registry bookkeeping
# would have to be stubbed out rather than ported. The four steps it performs
# are written out directly here instead - which is also easier to audit, and
# auditing is the point of this particular run.
#
# Prerequisites: 00_setup_env.sh and 01_prefetch.sh on a login node, and
# 02_push_adapters.sh from the laptop.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

MERGE="$MODELS_DIR/gemma3-xsum-merged-linear"
STUDENT="$MODELS_DIR/gemma3-xsum-kd-from-merge"

echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- 0. the alignment gate -------------------------------------------------
# Seconds, and it is the whole reason this run exists. A teacher==student KL
# that is not ~0 means the producer is still misplacing rows, and everything
# after this point would train on noise - silently, which is how the last month
# of KD numbers were generated.
echo "=== alignment check ==="
uv run --project evaluate python experiments/kd_alignment_smoke.py --examples 4 \
  | tee "$RESULTS_DIR/alignment_check.txt"
grep -qE 'one extra row.*0\.0000' "$RESULTS_DIR/alignment_check.txt" \
  || { echo "ALIGNMENT CHECK FAILED - refusing to train" >&2; exit 1; }

# --- 1. merge --------------------------------------------------------------
if [ ! -d "$MERGE" ]; then
  echo "=== merging the two half experts ==="
  cd "$REPO_ROOT/merge"
  uv run mergekit-yaml merge_linear_xsum_config.yaml "$MERGE" \
    --cuda --lazy-unpickle --lora-merge-cache "$REPO_ROOT/.lora-cache"
  cd "$REPO_ROOT"
else
  echo "=== merge already at $MERGE ==="
fi

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

# --- 2. teacher logprobs, each expert over its own half ---------------------
for i in 1 2; do
  OUT="$DATASETS_DIR/xsum_kd_half$i"
  if [ -f "$OUT/.kd_alignment_v2" ]; then
    echo "=== teacher $i already aligned at $OUT ==="
    continue
  fi
  rm -rf "$OUT"
  echo "=== teacher $i logprobs ==="
  uv run --project evaluate python precompute_logprobs.py \
    --model google/gemma-3-4b-it \
    --adapter "$MODELS_DIR/gemma3-xsum-$i-of-2-lora" \
    --dataset "$DATASETS_DIR/xsum_dataset$i/train" \
    --output "$OUT" --verify 32
  touch "$OUT/.precompute_complete"
done

if [ ! -d "$DATASETS_DIR/xsum_kd_train" ]; then
  echo "=== combining the two halves ==="
  uv run --project evaluate python concat_datasets.py \
    "$DATASETS_DIR/xsum_kd_half1" "$DATASETS_DIR/xsum_kd_half2" \
    --output "$DATASETS_DIR/xsum_kd_train"
fi

# --- 3. distil -------------------------------------------------------------
echo "=== training the student (kd_alpha 0.2 / kd_ce_alpha 1.0) ==="
cd "$REPO_ROOT/train"
uv run axolotl train xsum_gemma_kd_balanced.yaml
cd "$REPO_ROOT"

# --- 4. evaluate -----------------------------------------------------------
echo "=== evaluating ==="
cd "$REPO_ROOT/evaluate"
evaluate () {
  local name=$1 model=$2 adapter=${3:-}
  local args=(--model "$model" --dataset "$DATASETS_DIR/xsum_dataset/test"
              --output "$RESULTS_DIR/$name.json")
  [ -n "$adapter" ] && args+=(--adapter "$adapter")
  uv run python evaluate_summarisation.py "${args[@]}"
}

evaluate kd-aligned "$STUDENT"
evaluate merge      "$MERGE"
evaluate half1      google/gemma-3-4b-it "$MODELS_DIR/gemma3-xsum-1-of-2-lora"
evaluate half2      google/gemma-3-4b-it "$MODELS_DIR/gemma3-xsum-2-of-2-lora"
evaluate full       google/gemma-3-4b-it "$MODELS_DIR/gemma3-xsum-full-lora"

echo
echo "=== done. Results in $RESULTS_DIR ==="
echo "Compare against the pre-fix numbers (KD 0.3987, merge 0.3970, half 0.3990,"
echo "full 0.4049) with:"
echo "  uv run --project evaluate python experiments/bootstrap_rouge.py $RESULTS_DIR"
echo "  uv run --project evaluate python experiments/degeneracy.py $RESULTS_DIR"
