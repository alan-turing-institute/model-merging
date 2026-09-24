#!/bin/bash
#SBATCH --job-name=xsum-experts
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/expert-%A_%a.out
#SBATCH --error=logs/expert-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Retrain the three XSum experts on Isambard.
#
# WHY. The adapters copied from the laptop are a version-1-era generation, not
# the artefacts KD.md's numbers were measured against. All three produce 51-53
# word, 2.3-sentence output - the non-terminating signature - on the GH200, on
# the laptop, and identically. The correct versions (full-lora:5, halves:4,
# registered 14 September) exist in the Azure registry but cannot be fetched:
# the subscription is read-only and blob access is denied by RBAC.
#
# Retraining gives a SELF-CONSISTENT set instead of a bit-identical one. Every
# baseline, merge and KD student then sits on one platform with one generation
# of experts, which is what comparisons actually require. The data prep is
# deterministic - same seed, same 8000/500/1000 split - and the test set's id
# md5 was verified identical to the Azure one, so the evaluation target is
# unchanged even though the experts will not be bit-identical.
#
# NOT comparable, arm by arm, with KD.md's absolute numbers. Comparable within
# itself, which is the property the recovery fraction needs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs

case "$SLURM_ARRAY_TASK_ID" in
  1) CONFIG=xsum_gemma.yaml;  OUT=gemma3-xsum-full-lora ;;
  2) CONFIG=xsum_gemma1.yaml; OUT=gemma3-xsum-1-of-2-lora ;;
  3) CONFIG=xsum_gemma2.yaml; OUT=gemma3-xsum-2-of-2-lora ;;
  *) echo "unexpected array index $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "=== $CONFIG -> $OUT | $(hostname) | $(date -Is) ==="
grep -E '^eager_attention:|^load_in_4bit:|^num_epochs:|^learning_rate:' "$REPO_ROOT/train/$CONFIG"

if [ -f "$MODELS_DIR/$OUT/adapter_model.safetensors" ]; then
  echo "=== $OUT already trained, skipping ==="
  exit 0
fi

cd "$REPO_ROOT/train"
uv run axolotl train "$CONFIG"

# The failure this job exists to avoid is silent: a NaN run still saves an
# adapter, and the only symptom is an evaluation of nothing an hour later. Fail
# here instead, where the log still says why.
if grep -qiE "\bnan\b" "$REPO_ROOT/train/.axolotl-${CONFIG%.yaml}.log" 2>/dev/null; then
  echo "NaN seen in training - eager attention did not take. Refusing to mark complete." >&2
  exit 1
fi
echo "=== $OUT done $(date -Is) ==="
