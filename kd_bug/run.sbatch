#!/bin/bash
#SBATCH --job-name=kd-bug
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Usage: sbatch run.sbatch
set -euo pipefail

# The nodes' driver (580.173) now supports the cu130 wheels natively, and the
# older CUDA 13.0 forward-compat libcuda (580.65) breaks it with error 803. sbatch
# copies the submitting shell's LD_LIBRARY_PATH, so strip the compat dir from it.
LD_LIBRARY_PATH=$(tr ':' '
' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH

cd "$SLURM_SUBMIT_DIR"
PYTHONPATH=. uv run axolotl preprocess kd_readme_example.yaml
uv run python inspect_prepared.py
uv run python score_alignment.py
