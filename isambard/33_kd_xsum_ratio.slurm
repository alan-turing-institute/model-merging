#!/bin/bash
#SBATCH --job-name=xsum-kd-ratio
#SBATCH --array=1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/kd-ratio-%A_%a.out
#SBATCH --error=logs/kd-ratio-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# The kd_alpha sweep on the ORIGINAL two-way arm, in parallel.
#
# 30_kd_xsum.slurm re-runs 0.2/1.0, the one weighting KD.md still treats as
# standing. This adds the other two points. The question they settle is the one
# the alignment fix opened: the 0.9/0.1 arm degenerated badly - 41 empty and 55
# repeating out of 1000 - but it did so with a teacher scoring the token AFTER
# the one it described. Whether 0.9/0.1 damages a model when the teacher is
# correct has never been measured.
#
# Array rather than one job: the arms share a student and a teacher dataset and
# touch neither, so they are independent after the teacher pass. Two GPUs for an
# hour beats one GPU for two.
#
# DEPENDENCY. Both arms read ../datasets/xsum_kd_train, which 30_kd_xsum.slurm
# produces. Submit with:
#   sbatch --dependency=afterok:<30_kd_xsum jobid> isambard/31_kd_xsum_sweep.slurm
# Without it, three jobs race to write one dataset directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

# Teacher-dominant conditions at REAL ratios. Nominal 0.9/0.1 was ~1500:1 and
# collapsed; these are what weighting the teacher heavily should have meant.
case "$SLURM_ARRAY_TASK_ID" in
  1) TAG=t9 ;;   # 9:1 teacher:CE per token, kd_alpha 0.0545
  2) TAG=t3 ;;   # 3:1, kd_alpha 0.0182
  *) echo "unexpected array index" >&2; exit 1 ;;
esac
CONFIG="xsum_gemma_kd_${TAG}.yaml"
STUDENT="$MODELS_DIR/gemma3-xsum-kd-${TAG}"

echo "=== $TAG | $(hostname) | $(date -Is) ==="
grep -E '^kd_alpha:|^kd_ce_alpha:' "$REPO_ROOT/train/$CONFIG"

# The teacher data must exist AND carry the alignment marker. A directory
# written before 6f923a1 trains happily and teaches noise, which is the whole
# reason this sweep exists.
[ -f "$DATASETS_DIR/xsum_kd_train/.kd_alignment_v2" ] || {
  echo "teacher data missing or pre-fix at $DATASETS_DIR/xsum_kd_train" >&2
  echo "run 30_kd_xsum.slurm first, or submit this with --dependency=afterok" >&2
  exit 1
}

cd "$REPO_ROOT/train"
# axolotl writes a LoRA adapter, so a rerun that retrains is pure waste - and
# these jobs have already been resubmitted twice for failures downstream of
# training.
# Newer than the merge, not merely present. A stale adapter trained against a
# PREVIOUS merge is silently reusable otherwise: the ctx arm skipped training on
# an adapter from the day before and evaluated it attached to a merge it had
# never seen. Existence is not freshness - the same mistake as testing for the
# concat directory instead of its alignment marker.
if [ -f "$STUDENT/adapter_model.safetensors" ] \
   && [ "$STUDENT/adapter_model.safetensors" -nt "$MODELS_DIR/gemma3-xsum-merged-linear/config.json" ]; then
  echo "=== student already trained against this merge at $STUDENT, skipping ==="
else
  uv run axolotl train "$CONFIG"
fi
cd "$REPO_ROOT/evaluate"
# The student is a LoRA adapter on the merge, not a standalone model.
uv run python evaluate_summarisation.py \
  --model "$MODELS_DIR/gemma3-xsum-merged-linear" \
  --adapter "$STUDENT" \
  --dataset "$DATASETS_DIR/xsum_dataset/test" \
  --output "$RESULTS_DIR/kd-${TAG}.json"

echo "=== $TAG done $(date -Is) ==="
