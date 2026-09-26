#!/bin/bash
#SBATCH --job-name=mistral-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/mistral-smoke-%j.out
#SBATCH --error=logs/mistral-smoke-%j.out
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

OUT="$MODELS_DIR/mistral-crime-full-lora"
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
  uv run axolotl train crime_mistral.yaml
  cd "$REPO_ROOT"
fi

TRAIN_LOG="${SLURM_SUBMIT_DIR:-$PWD}/logs/mistral-smoke-${SLURM_JOB_ID}.out"
if grep -qiE "\bnan\b" "$TRAIN_LOG"; then
  echo "NaN during training - refusing to report this expert as usable" >&2; exit 1
fi

# The check Qwen3 lacked: does it emit the label, and stop?
cd "$REPO_ROOT/evaluate"
for T in 5 16 48; do
  echo "=== generation budget $T ==="
  uv run python evaluate.py --model "$MISTRAL_DIR" \
    --adapter "$OUT" --dataset "$DATASETS_DIR/crime_dataset/test" \
    --limit 80 --max-new-tokens $T \
    --output "$RESULTS_DIR/mistral-crime-t${T}.json" 2>&1 \
    | grep -E "^Accuracy|^Unparsed" || true
done

cd "$REPO_ROOT"
python3 - <<'PY'
import json, os
print()
print("budget  accuracy  unparsed  first prediction")
for t in (5, 16, 48):
    f = os.path.join(os.environ["RESULTS_DIR"], "mistral-crime-t%d.json" % t)
    if not os.path.exists(f):
        print("%-7d (missing)" % t); continue
    d = json.load(open(f)); m = d["metrics"]
    print("%-7d %-9.4f %-9s %r" % (t, m.get("accuracy", 0), m.get("num_unparsed", "?"),
                                   str(d.get("predictions", [""])[0])[:50]))
print()
print("Reference: the crime test set is 44.3% positive, so ~0.443 means the")
print("model is answering one class regardless - which is how Qwen3 failed.")
PY
echo "=== smoke complete $(date -Is) ==="
