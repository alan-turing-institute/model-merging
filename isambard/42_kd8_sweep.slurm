#!/bin/bash
#SBATCH --job-name=xsum-kd8-sweep
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/kd8-sweep-%A_%a.out
#SBATCH --error=logs/kd8-sweep-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# The kd_alpha / kd_ce_alpha sweep, one arm per array task, all three sharing
# the student (the 8-way merge) and the teacher data (40_kd8_teachers.slurm).
# They differ in two scalars and nothing else.
#
#   task 1   0.9 / 0.1   axolotl's default; the setting that degenerated
#   task 2   0.5 / 0.5   the midpoint
#   task 3   0.2 / 1.0   measured-magnitude weighting; KD.md's standing number
#
# The question, stated in KD.md's 2026-09-24 section: the 0.9/0.1 -> 0.2/1.0
# improvement was real, but it was measured with a teacher scoring the wrong
# token. Whether a correctly aligned teacher at 0.9/0.1 also damages the model
# is untested, and these three arms are the test.
#
#   sbatch --dependency=afterok:<concat-job-id> isambard/42_kd8_sweep.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

SPLITS="${SPLITS:-8}"
TAGS=(a09 a05 a02)
TAG="${TAGS[$((SLURM_ARRAY_TASK_ID - 1))]}"
CONFIG="xsum_gemma_kd${SPLITS}_${TAG}.yaml"
STUDENT="$MODELS_DIR/gemma3-xsum-kd${SPLITS}-${TAG}"
NAME="kd${SPLITS}-${TAG}"

echo "=== $NAME | $CONFIG | $(hostname) | $(date -Is) ==="
# Diagnostic, never a precondition: under `set -e` a failing nvidia-smi
# aborts the job in under a second, before anything is logged - which is
# exactly how the crime chain died with an empty output file.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo '  (nvidia-smi unavailable on this node)'
grep -hE '^(kd_alpha|kd_ce_alpha):' "$REPO_ROOT/train/$CONFIG"

# The concat step verified every shard carried .kd_alignment_v2; check the
# combined set exists rather than re-deriving that here.
[ -f "$DATASETS_DIR/xsum_kd${SPLITS}_train/.kd_alignment_v2" ] \
  || { echo "teacher data missing - run 41_kd8_concat.slurm first" >&2; exit 1; }

cd "$REPO_ROOT/train"
uv run axolotl train "$CONFIG"

echo "=== evaluating $NAME $(date -Is) ==="
cd "$REPO_ROOT/evaluate"
uv run python evaluate_summarisation.py \
  --model "$STUDENT" \
  --dataset "$DATASETS_DIR/xsum_dataset/test" \
  --output "$RESULTS_DIR/$NAME.json"

echo "=== $NAME done $(date -Is) ==="
