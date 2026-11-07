#!/bin/bash
#SBATCH --job-name=moe-routing
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
#SBATCH --output=../evaluate/slurm/%x-%j.out
# Usage, from merge/: sbatch moe_routing.sbatch <moe_routing.py arguments>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
LD_LIBRARY_PATH=$(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH
../evaluate/.venv/bin/python moe_routing.py "$@"
