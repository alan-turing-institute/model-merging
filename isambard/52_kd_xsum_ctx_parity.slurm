#!/bin/bash
#SBATCH --job-name=xsum-ctxkd-parity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/ctxkd-parity-%j.out
#SBATCH --error=logs/ctxkd-parity-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Context distillation at kd_alpha 0.9 / kd_ce_alpha 1.0.
#
# The second point on the teacher-weight axis for the ctx arm. 50_kd_xsum_ctx
# runs 0.2/1.0; this raises the teacher term by ~4.5x with cross-entropy held at
# full weight, which is the comparison 0.9/0.1 could never give - that setting
# moved both weights at once, so its degeneracy was never attributable to either.
#
# DEPENDENCY. Reads ../datasets/xsum_ctxkd_train, which 50_kd_xsum_ctx builds.
# Submit with:
#   sbatch --dependency=afterok:<50_kd_xsum_ctx jobid> isambard/51_kd_xsum_ctx_a09.slurm
# Two jobs producing that directory at once would corrupt it.
#
# Baselines come from the 50 job, which measures merge/halves/full on the same
# ctx test set; this trains and evaluates one student only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

STUDENT="$MODELS_DIR/gemma3-xsum-ctxkd-parity"

echo "=== $(hostname) | $(date -Is) ==="
grep -E '^kd_alpha:|^kd_ce_alpha:' "$REPO_ROOT/train/xsum_gemma_ctxkd_parity.yaml"

[ -f "$DATASETS_DIR/xsum_ctxkd_train/.kd_alignment_v2" ] || {
  echo "ctx teacher data missing or pre-fix at $DATASETS_DIR/xsum_ctxkd_train" >&2
  echo "run 50_kd_xsum_ctx.slurm first, or submit with --dependency=afterok" >&2
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
  uv run axolotl train xsum_gemma_ctxkd_parity.yaml
fi
cd "$REPO_ROOT/evaluate"
# The student is a LoRA adapter on the merge, not a standalone model.
uv run python evaluate_summarisation.py \
  --model "$MODELS_DIR/gemma3-xsum-merged-linear" \
  --adapter "$STUDENT" \
  --dataset "$DATASETS_DIR/xsum_ctx_dataset/test" \
  --output "$RESULTS_DIR/ctx-kd-parity.json"

echo "=== done $(date -Is) ==="
