#!/bin/bash
#SBATCH --job-name=base-and-cost
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Two things the feasibility report asserts but has never measured.
#
# B. THE BASE MODEL ON BOTH TASKS, on the SAME 200 examples as the cross-task
#    merge and the oracle router (45_crosstask_merge_eval.slurm, 44_adapter_
#    routing.slurm). Cross-task retention is currently reported as a raw ratio,
#    merged/expert = 98%. The project's other retention figure, and CROSSTASK.md's
#    own definition, subtract the base: (merged - base)/(expert - base), which
#    asks what share of the ability FINE-TUNING ADDED survives merging. That
#    cannot be computed today because the base has never been scored on these
#    200. Both readings are defensible; the report should state which it means
#    and be able to give the other.
#
# C. THE COST OF SERVING BOTH SKILLS. "At no extra inference cost" is an
#    architectural argument, not a measurement. experiments/inference_cost.py
#    compares the merge against the routing alternative on one GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

BASE=google/gemma-3-4b-it
MERGE="$MODELS_DIR/gemma3-crosstask-taskarith"
echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

cd "$REPO_ROOT/evaluate"
echo "=== B1: base model on crime, same 200 as the merge and the router ==="
uv run python evaluate.py --model "$BASE" \
  --dataset "$DATASETS_DIR/crime_dataset/test" --limit 200 \
  --output "$RESULTS_DIR/crosstask-base-crime.json" 2>&1 | grep -E "^Accuracy|^Unparsed" || true

echo "=== B2: base model on xsum, same 200 ==="
uv run python evaluate_summarisation.py --model "$BASE" \
  --dataset "$DATASETS_DIR/xsum_dataset/test" --limit 200 \
  --output "$RESULTS_DIR/crosstask-base-xsum.json" 2>&1 | grep -E "^rouge1|Length" || true

cd "$REPO_ROOT"
echo "=== C: inference cost, merge against the routing alternative ==="
uv run --project evaluate python experiments/inference_cost.py \
  --base "$BASE" --merged "$MERGE" \
  --crime-adapter "$MODELS_DIR/gemma3-crime-full-lora" \
  --xsum-adapter "$MODELS_DIR/gemma3-xsum-full-lora" \
  --crime-dataset "$DATASETS_DIR/crime_dataset/test" \
  --xsum-dataset "$DATASETS_DIR/xsum_dataset/test" \
  --output "$RESULTS_DIR/inference-cost.json"
echo "=== done $(date -Is) ==="
