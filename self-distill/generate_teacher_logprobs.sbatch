#!/bin/bash
#SBATCH --job-name=teacher-logprobs
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Usage, from self-distill/:
#   sbatch generate_teacher_logprobs.sbatch <model> <dataset_dir> <output_dir> [prefix] [extra args...]
#   e.g. sbatch generate_teacher_logprobs.sbatch google/gemma-3-4b-it \
#            $PROJECTDIR/$USER/model-merging/datasets/xsum_prompts \
#            $PROJECTDIR/$USER/model-merging/datasets xsum_kd_data
#        sbatch generate_teacher_logprobs.sbatch google/gemma-3-4b-it \
#            ../datasets/xsum_prompts_1_of_3 ../datasets xsum_kd_data_1_of_3 --top-k 50
#
# model is a Hugging Face hub id or a local directory. prefix defaults to the
# script's own default (kd_data); the datasets are written to
# <output_dir>/<prefix>_long and <output_dir>/<prefix>_short. Anything after
# prefix goes to generate_teacher_logprobs.py unchanged (see its --help); to pass
# extra args without a prefix, give "" as the prefix. Relative paths resolve
# against self-distill/.
set -euo pipefail
USAGE="usage: sbatch generate_teacher_logprobs.sbatch <model> <dataset_dir> <output_dir> [prefix] [extra args...]"
MODEL=${1:?$USAGE}
DATASET_DIR=${2:?$USAGE}
OUTPUT_DIR=${3:?$USAGE}
PREFIX=${4:-}
shift $(( $# < 4 ? $# : 4 ))

# Submit from self-distill/: uv needs its pyproject.toml to find the environment.
cd "$SLURM_SUBMIT_DIR"
[[ -f generate_teacher_logprobs.py ]] || { echo "no generate_teacher_logprobs.py in $PWD - submit from self-distill/" >&2; exit 1; }
[[ -d $DATASET_DIR ]] || { echo "$DATASET_DIR not found" >&2; exit 1; }

PREFIX_ARGS=()
[[ -n $PREFIX ]] && PREFIX_ARGS=(--prefix "$PREFIX")

# Required in every job script on Isambard-AI. Without it the cu130 torch build
# cannot see the GPU.
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Hugging Face downloads (the teacher model) live in the project area, not the
# 100 GB home quota. See the main README.
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is the CUDA compat lib on LD_LIBRARY_PATH?'"

uv run python generate_teacher_logprobs.py "$MODEL" "$DATASET_DIR" "$OUTPUT_DIR" \
  "${PREFIX_ARGS[@]}" "$@"
