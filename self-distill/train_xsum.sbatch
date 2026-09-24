#!/bin/bash
#SBATCH --job-name=xsum-kd
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=06:00:00
#SBATCH --output=../models/%x-%j.out
#SBATCH --error=../models/%x-%j.out
# Usage: sbatch train_xsum.sbatch long    (or: short)
#
set -euo pipefail
VARIANT=${1:?usage: sbatch train_xsum.sbatch <long|short>}

# Required in every job script. Without it torch silently falls back to CPU --
# Axolotl does not error, it just sets device: cpu and grinds.
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd "$SLURM_SUBMIT_DIR"
uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is the CUDA compat lib on LD_LIBRARY_PATH?'"

# Preprocess separately, as Axolotl itself recommends -- it keeps the
# memory-hungry tokenisation pass out of the training process.
uv run axolotl preprocess "self-distillation_xsum_${VARIANT}.yaml"
uv run axolotl train      "self-distillation_xsum_${VARIANT}.yaml"
