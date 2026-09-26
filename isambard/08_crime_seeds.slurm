#!/bin/bash
#SBATCH --job-name=crime-seeds
#SBATCH --array=1-10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/crime-seed-%A_%a.out
#SBATCH --error=logs/crime-seed-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Five seeds of the crime parity arm, paired with five of the no-teacher
# control at the same seeds.
#
# WHY. The single-run result - distilled 0.9721 against a merge of 0.9638, R =
# +0.72 - is an improvement of 0.0083, and this task's measured seed noise is
# sd 0.0078 over a 0.9508-0.9721 range. The distilled run scored the top of
# that range exactly. One draw from a distribution that wide is how this
# project's previous headline (+0.78) was produced, and five seeds are what
# dissolved it.
#
# Paired by construction: arm and control share a seed, a merge, a teacher
# dataset and a budget, and differ only in kd_alpha. That licenses a paired
# test on the five differences rather than comparing two independent means.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

i="$SLURM_ARRAY_TASK_ID"
if [ "$i" -le 5 ]; then ARM=distilled; SEED=$i; else ARM=control; SEED=$((i - 5)); fi
CONFIG="crime_kd_${ARM}_s${SEED}.yaml"
STUDENT="$MODELS_DIR/gemma3-crime-kd-${ARM}-s${SEED}"
MERGE="$MODELS_DIR/gemma3-crime-merged-linear-2"

echo "=== crime $ARM seed $SEED | $(hostname) | $(date -Is) ==="
grep -E '^seed:|^kd_alpha:|^kd_ce_alpha:' "$REPO_ROOT/train/$CONFIG"

[ -f "$MERGE/config.json" ] || { echo "merge missing at $MERGE" >&2; exit 1; }
[ -f "$DATASETS_DIR/crime_kd_train/.kd_alignment_v2" ] || {
  echo "teacher data missing or pre-fix at $DATASETS_DIR/crime_kd_train" >&2; exit 1; }

if [ -f "$STUDENT/adapter_model.safetensors" ] \
   && [ "$STUDENT/adapter_model.safetensors" -nt "$MERGE/config.json" ]; then
  echo "=== already trained against this merge, skipping ==="
else
  cd "$REPO_ROOT/train"
  uv run axolotl train "$CONFIG"
  cd "$REPO_ROOT"
fi

cd "$REPO_ROOT/evaluate"
uv run python evaluate.py \
  --model "$MERGE" --adapter "$STUDENT" \
  --dataset "$DATASETS_DIR/crime_dataset/test" \
  --output "$RESULTS_DIR/crime-${ARM}-s${SEED}.json"

echo "=== crime $ARM seed $SEED done $(date -Is) ==="
