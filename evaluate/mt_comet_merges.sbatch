#!/bin/bash
#SBATCH --job-name=mt-comet-merges
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100000M
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# COMET-22 on every EN->DE run (merges, experts, task-parallel, distilled):
# export from the .eval logs here, score in the metrics venv (unbabel-comet pins
# transformers 4.x). See mt_comet_merges.py.
#   cd evaluate && sbatch mt_comet_merges.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
# TMPDIR is /local/user/<uid> -> /run/user/<uid>, which systemd removes when the
# user's last session on the node ends; a COMET run with a file mapped there then
# dies with a bus error (twice for summaries, once for the replication). Use a
# job-private directory instead (as summ_metrics.sbatch does).
export TMPDIR=$PROJECTDIR/$USER/mm/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECTDIR/$USER/mm/hf-cache}
METRICS_VENV=${METRICS_VENV:-$PROJECTDIR/$USER/mm/venvs/metrics}
OUT=../results/mt_comet_merge_inputs
uv run --no-sync python mt_comet_export.py logs "$OUT" --all
"$METRICS_VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no GPU'"
"$METRICS_VENV/bin/python" mt_comet_merges.py "$OUT"
