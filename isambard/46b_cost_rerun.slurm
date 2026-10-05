#!/bin/bash
#SBATCH --job-name=cost-rerun
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Experiment C only, re-run with interleaved repeats. The first pass ran each
# configuration once in sequence and charged the GPU's clock ramp to whichever
# went first, which put the base model 2.4x below a merge of identical
# architecture. B is already measured and is not repeated here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader || true
uv run --project evaluate python experiments/inference_cost.py \
  --base google/gemma-3-4b-it --merged "$MODELS_DIR/gemma3-crosstask-taskarith" \
  --crime-adapter "$MODELS_DIR/gemma3-crime-full-lora" \
  --xsum-adapter "$MODELS_DIR/gemma3-xsum-full-lora" \
  --crime-dataset "$DATASETS_DIR/crime_dataset/test" \
  --xsum-dataset "$DATASETS_DIR/xsum_dataset/test" \
  --repeats 3 --output "$RESULTS_DIR/inference-cost.json"
echo "=== done $(date -Is) ==="
