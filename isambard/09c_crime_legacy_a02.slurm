#!/bin/bash
#SBATCH --job-name=crime-legacy-a02
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/crime-legacy-a02-%A_%a.out
#SBATCH --error=logs/crime-legacy-a02-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# The misaligned teacher at 0.2 / 1.0 - the weighting the Azure arm ran at when
# it reported 0.9707 +/- 0.0011 from a teacher that was ALSO misaligned.
#
# WHY. 09b reproduced the misaligned arm at 0.0625 / 1.0 and got 0.9214 +/-
# 0.0327: far below the merge, and the noisiest arm this project has measured.
# The Azure figure is the opposite on both counts, and the two runs differ in
# weighting as well as generation. 0.2 / 1.0 is roughly 3:1 per token against
# 0.0625's 1:1, so MORE weight on a signal that is pure noise should be worse,
# not better. This cell holds the generation fixed and tests that directly.
#
# Read against results-isambard/crime-legacy-s*.json (same teacher data, same
# seeds, same merge; kd_alpha is the only difference) and against
# crime-distilled-s*.json (the aligned arm at 0.0625).
#
# Depends on the teacher data from 09a. Run that first if it is not on disk.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

SEED="$SLURM_ARRAY_TASK_ID"
CONFIG="crime_kd_legacy_a02_s${SEED}.yaml"
STUDENT="$MODELS_DIR/gemma3-crime-kd-legacy-a02-s${SEED}"
MERGE="$MODELS_DIR/gemma3-crime-merged-linear-2"

echo "=== crime legacy-teacher a02 seed $SEED | $(hostname) | $(date -Is) ==="
grep -E '^seed:|^kd_alpha:|^kd_ce_alpha:|^output_dir:' "$REPO_ROOT/train/$CONFIG"
grep -E 'crime_kd_train' "$REPO_ROOT/train/$CONFIG"

[ -f "$MERGE/config.json" ] || { echo "merge missing at $MERGE" >&2; exit 1; }
# The weighting is the whole point of this arm, so assert it rather than trust
# the generator: a copy that silently kept 0.0625 would duplicate 09b and read
# as a clean replication.
grep -qx 'kd_alpha: 0.2' "$REPO_ROOT/train/$CONFIG" \
  || { echo "$CONFIG is not at kd_alpha 0.2 - refusing" >&2; exit 1; }
grep -qx 'kd_ce_alpha: 1.0' "$REPO_ROOT/train/$CONFIG" \
  || { echo "$CONFIG is not at kd_ce_alpha 1.0 - refusing" >&2; exit 1; }

[ -f "$DATASETS_DIR/crime_kd_train_legacy/.kd_alignment_legacy" ] || {
  echo "pre-fix teacher data missing or unmarked at $DATASETS_DIR/crime_kd_train_legacy" >&2
  echo "run isambard/09a_crime_legacy_teacher.slurm first" >&2; exit 1; }
[ ! -f "$DATASETS_DIR/crime_kd_train_legacy/.kd_alignment_v2" ] || {
  echo "that directory carries the ALIGNED marker too - refusing" >&2; exit 1; }

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
  --output "$RESULTS_DIR/crime-legacy-a02-s${SEED}.json"

echo "=== crime legacy a02 seed $SEED done $(date -Is) ==="
