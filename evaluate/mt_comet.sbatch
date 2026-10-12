#!/bin/bash
#SBATCH --job-name=mt-comet
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
# COMET-22 for the translation runs: export triples from the .eval logs in this
# environment, then score them in the metrics venv, which unbabel-comet needs
# (it pins transformers back to 4.x).
#
#   cd evaluate && sbatch mt_comet.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
export HF_HUB_OFFLINE=1
METRICS_VENV=${METRICS_VENV:-$PROJECTDIR/$USER/mm/venvs/metrics}
OUT=../results/mt_comet_inputs

uv run --no-sync python mt_comet_export.py logs "$OUT"
"$METRICS_VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no GPU'"
"$METRICS_VENV/bin/python" mt_comet_score.py "$OUT"
