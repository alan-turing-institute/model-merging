#!/bin/bash
#SBATCH --job-name=comet22-replication
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Replicate COMET-22's WMT21 MQM correlations (comet22_replication.py). From evaluate/:
#   sbatch comet22_replication.sbatch
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
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECTDIR/$USER/mm/hf-cache}
METRICS_VENV=${METRICS_VENV:-$PROJECTDIR/$USER/mm/venvs/metrics}
OUT=../results/comet22_replication
(cd ../train && uv run --no-sync python ../evaluate/comet22_replication.py export "$OUT")
HF_HUB_OFFLINE=1 "$METRICS_VENV/bin/python" comet22_replication.py score "$OUT"
