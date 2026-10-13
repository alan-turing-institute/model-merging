#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
# Untrained-model pilot for the constrained-summary and turn-reorder tasks.
#   sbatch dialogue_pilot.sbatch google/gemma-3-27b-it 8     # model, best-of k
# Scores every model generated so far, so the last job to finish prints the table.
set -euo pipefail
MODEL=${1:?usage: dialogue_pilot.sbatch <model> [best_of]}
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
export UV_PROJECT_ENVIRONMENT=${SELF_DISTILL_VENV:-$PROJECTDIR/$USER/mm/venvs/self-distill}
export HF_HUB_OFFLINE=1
OUT=../results/dialogue_pilot
uv run python dialogue_pilot.py "$MODEL" ../datasets/samsum_raw "$OUT" --best-of "${2:-0}"
../evaluate/.venv/bin/python dialogue_tasks.py score "$OUT"
