#!/bin/bash
#SBATCH --job-name=adapter-routing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:30:00
#SBATCH --output=logs/routing-%j.out
#SBATCH --error=logs/routing-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Per-example adapter routing against merging, on the cross-task pair.
#
# This is the sparse-MoE question asked on the Gemma-3 experts every other
# result uses. mergekit-moe emits no Gemma architecture, and the Qwen3 port
# built to work around that produced experts that do not answer their task at
# generation time - so routing between adapters is the tractable form.
#
# The expected effect is small and the bound is already known: routing cannot
# beat each expert on its own task, and the cross-task sweep measured merging
# at 99.6% and 98.6% of that. This converts those ratios into absolute costs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader \
  || echo "  (nvidia-smi unavailable on this node)"

uv run --project evaluate python experiments/adapter_routing.py \
  --crime-adapter "$MODELS_DIR/gemma3-crime-full-lora" \
  --xsum-adapter  "$MODELS_DIR/gemma3-xsum-full-lora" \
  --crime-data    "$DATASETS_DIR/crime_dataset/test" \
  --xsum-data     "$DATASETS_DIR/xsum_dataset/test" \
  --limit 200 \
  --output "$RESULTS_DIR/adapter-routing.json"

echo "=== done $(date -Is) ==="
