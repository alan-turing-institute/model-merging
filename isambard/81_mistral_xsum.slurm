#!/bin/bash
#SBATCH --job-name=mistral-xsum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/mistral-xsum-%j.out
#SBATCH --error=logs/mistral-xsum-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# ONE expert, then verify it before anything is built on it.
#
# The Qwen3 port went the other way: six experts trained and a sparse MoE built
# before a single expert was evaluated. They had converged - the crime expert
# reached loss 0.12 - and still scored the always-positive baseline at
# generation, because converging on a teacher-forced target says nothing about
# whether the model emits that target and stops. Roughly two hours of GPU went
# into an arm whose parents were broken.
#
# So this trains the crime expert only and reports, at three generation
# budgets, what it actually emits. Nothing else is queued until that is read.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

OUT="$MODELS_DIR/mistral-xsum-full-lora"
# The cached snapshot omits consolidated.safetensors - Mistral's own single-file
# format, which transformers never reads but snapshot_download's completeness
# check rejects. Resolve the directory directly so the repo id is never looked up.
MISTRAL_DIR=$(ls -d "$HF_HUB_CACHE"/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/ | head -1)
echo "base: $MISTRAL_DIR"
[ -f "$MISTRAL_DIR/config.json" ] || { echo "Mistral snapshot not found" >&2; exit 1; }
echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || echo "  (nvidia-smi unavailable on this node)"

if [ -f "$OUT/adapter_model.safetensors" ]; then
  echo "=== expert already trained ==="
else
  cd "$REPO_ROOT/train"
  uv run axolotl train xsum_mistral.yaml
  cd "$REPO_ROOT"
fi

TRAIN_LOG="${SLURM_SUBMIT_DIR:-$PWD}/logs/mistral-xsum-${SLURM_JOB_ID}.out"
if grep -qiE "\bnan\b" "$TRAIN_LOG"; then
  echo "NaN during training - refusing to report this expert as usable" >&2; exit 1
fi

# The same check the crime expert passed, in the form this task needs. For
# summarisation the failure Qwen3 showed would appear as runaway length rather
# than unparsed labels: a model that never emits end-of-turn generates to the
# cap. Gemma's XSum experts produce 19.1-19.4 words against a 21-word
# reference; anything near the 64-token cap means it is not terminating.
cd "$REPO_ROOT/evaluate"
uv run python evaluate_summarisation.py --model "$MISTRAL_DIR" \
  --adapter "$OUT" --dataset "$DATASETS_DIR/xsum_dataset/test" \
  --limit 200 --output "$RESULTS_DIR/mistral-xsum-full.json" 2>&1 \
  | grep -E "^rouge1|^semantic|Length|^Examples" || true

cd "$REPO_ROOT"
python3 - <<'PY'
import json, os
f = os.path.join(os.environ["RESULTS_DIR"], "mistral-xsum-full.json")
d = json.load(open(f)); m = d["metrics"]
print()
print("ROUGE-1 %.4f   words %.1f   sentences %.2f   empty %d" % (
    m.get("rouge1", 0), m.get("mean_pred_words", 0),
    m.get("mean_pred_sentences", 0), m.get("num_empty", 0)))
print("first prediction: %r" % str(d.get("predictions", [""])[0])[:110])
print()
print("Reference is 21.0 words in one sentence. Gemma's XSum experts land at")
print("19.1-19.4. Near the 64-token cap means the model is not terminating,")
print("which is how the Qwen3 port failed.")
PY
echo "=== xsum expert complete $(date -Is) ==="
