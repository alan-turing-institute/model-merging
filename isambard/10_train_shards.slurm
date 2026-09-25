#!/bin/bash
#SBATCH --job-name=xsum-shards
#SBATCH --array=1-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/shard-%A_%a.out
#SBATCH --error=logs/shard-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# One shard expert per array task. At 8 shards each sees 1,000 examples, about
# 125 steps - roughly 15 minutes on a GH200 - so the array finishes in the time
# one A100 run used to take.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs

SPLITS="${SPLITS:-8}"
i="$SLURM_ARRAY_TASK_ID"
CONFIG="xsum_gemma_s${i}_of_${SPLITS}.yaml"

echo "=== shard $i of $SPLITS | $(hostname) | $(date -Is) ==="
# Diagnostic, never a precondition: under `set -e` a failing nvidia-smi
# aborts the job in under a second, before anything is logged - which is
# exactly how the crime chain died with an empty output file.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo '  (nvidia-smi unavailable on this node)'

cd "$REPO_ROOT/train"
uv run axolotl train "$CONFIG"

# axolotl writes the adapter under output_dir; mark completion explicitly so a
# resubmitted array can tell a finished shard from one killed at the time limit.
touch "$MODELS_DIR/gemma3-xsum-${i}-of-${SPLITS}-lora/.training_complete"
echo "=== shard $i done $(date -Is) ==="
