#!/bin/bash
#SBATCH --job-name=mt-headroom
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
# Per-dataset translation headroom (mt_headroom.py): chrF and BLEU from the logs
# in this environment, then COMET-22 in the metrics venv.
#
#   cd evaluate && sbatch mt_headroom.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
export HF_HUB_OFFLINE=1
METRICS_VENV=${METRICS_VENV:-$PROJECTDIR/$USER/mm/venvs/metrics}
OUT=../results/mt_headroom_inputs

uv run --no-sync python mt_headroom.py logs "$OUT"
"$METRICS_VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no GPU'"
"$METRICS_VENV/bin/python" mt_comet_score.py "$OUT" mt-headroom-comet.json
