#!/bin/bash
#SBATCH --job-name=metric-bakeoff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/metric-bakeoff-%j.out
#SBATCH --error=logs/metric-bakeoff-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Which summarisation metric notices when a summary is WRONG?
#
# Seven metrics on the same perturbations of the same references, where the
# ground truth holds by construction. Two truths are tracked separately because
# the metrics answer different questions: does the candidate say what the
# reference says, and is it supported by the source article. A similarity metric
# scoring a hallucinated name highly is not malfunctioning - it is answering the
# first question - but SOMETHING in the suite has to answer the second, and this
# measures which does.
#
# Needs roberta-large and deberta-large-mnli in $HF_HUB_CACHE, prefetched from a
# login node: compute nodes have no outbound network.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no nvidia-smi)"

for f in merge.json full.json; do
  [ -f "$RESULTS_DIR/$f" ] || { echo "missing $RESULTS_DIR/$f" >&2; exit 1; }
done
for m in models--FacebookAI--roberta-large models--microsoft--deberta-large-mnli; do
  [ -d "$HF_HUB_CACHE/$m" ] || { echo "not prefetched: $m" >&2; exit 1; }
done

uv run --project evaluate python experiments/metric_bakeoff.py \
  --results "$RESULTS_DIR/merge.json" \
  --second  "$RESULTS_DIR/full.json" \
  --dataset "$DATASETS_DIR/xsum_dataset/test" \
  --output  "$RESULTS_DIR/metric-bakeoff.json" \
  --limit 200
