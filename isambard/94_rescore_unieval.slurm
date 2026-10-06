#!/bin/bash
#SBATCH --job-name=rescore-unieval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Re-score the stored XSum generations under UniEval's four dimensions.
#
# The metric validation ranked UniEval's dimensions first, second, third and
# fifth of nine for separating an intact summary from a corrupted one, and the
# re-scoring in 93_rescore_xsum.slurm then used none of them - it covers ROUGE-1,
# cosine and BERTScore, which placed seventh, eighth and ninth. This runs the
# same 21 arms, the same 200 articles and the same paired bootstrap (seed 0,
# 2,000 resamples, 'merge' as baseline) so the two tables line up row for row.
#
# Nothing is regenerated: the predictions are already on disk.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

# Identical list to 93_rescore_xsum.slurm, including the two degenerate arms
# (model_stock, kd2-a09). Those are the controls: a battery that cannot separate
# them from the merge is not measuring anything, and under the previous seven
# metrics both separated on 7 of 7.
ARMS="full half1 half2 merge
      kd-parity kd-t3 kd-t9 kd-aligned kd2-a09
      sweep-xsum-linear sweep-xsum-slerp sweep-xsum-task_arithmetic
      sweep-xsum-dare_ties sweep-xsum-ties sweep-xsum-arcee_fusion sweep-xsum-model_stock
      merge-4way shard1-of-4 kd4-parity
      ctx-merge ctx-kd-parity"

echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

uv run --project evaluate python experiments/rescore_xsum_unieval.py \
  --results-dir "$RESULTS_DIR" \
  --arms $ARMS \
  --dataset "$DATASETS_DIR/xsum_dataset/test" \
  --output  "$RESULTS_DIR/xsum-rescored-unieval.json" \
  --limit 200
echo "=== done $(date -Is) ==="
