#!/bin/bash
#SBATCH --job-name=semantic-validation
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/semantic-validation-%j.out
#SBATCH --error=logs/semantic-validation-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Validate the semantic metric rather than use it.
#
# WHY. Every summarisation arm reports a semantic score, and it is read as a
# check on whether a ROUGE difference reflects meaning or only wording. That
# reading assumes the metric penalises a summary that is WRONG, which nothing
# in this project has tested - and semantic_metrics.py own docstring doubts it,
# recording that swapping Mick Lally for Sean Lally at 73 scores HIGHER than
# ROUGE does. This perturbs references in ways whose ground truth is known and
# reads all three metrics on each.
#
# No GPU: MiniLM over a few thousand short sentences is seconds of CPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

[ -f "$RESULTS_DIR/merge.json" ] || { echo "need merge.json for predictions" >&2; exit 1; }
[ -f "$RESULTS_DIR/full.json" ]  || { echo "need full.json for the paraphrase pair" >&2; exit 1; }

uv run --project evaluate python experiments/semantic_validation.py \
  --results "$RESULTS_DIR/merge.json" \
  --second  "$RESULTS_DIR/full.json" \
  --dataset "$DATASETS_DIR/xsum_dataset/test" \
  --output  "$RESULTS_DIR/semantic-validation.json"
