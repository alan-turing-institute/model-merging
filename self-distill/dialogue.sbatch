#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
# Dialogue-task stages, run from self-distill/:
#   sbatch dialogue.sbatch teacher <constrained|constrained2> <model> <out.jsonl> [best_of] [adapter]
#   sbatch dialogue.sbatch eval <base_model> <curve> <name> [adapter]
# <curve> is reorder, constrained, constrained2, or a variant such as constrained2_sft: the
# task is the part before the first underscore.
set -euo pipefail
STAGE=${1:?usage: dialogue.sbatch <teacher|eval> ...}
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
export UV_PROJECT_ENVIRONMENT=${SELF_DISTILL_VENV:-$PROJECTDIR/$USER/mm/venvs/self-distill}
export HF_HUB_OFFLINE=1
case $STAGE in
  teacher)
    uv run python dialogue_data.py teacher "$2" "$3" ../datasets/samsum_raw "$4" --best-of "${5:-8}" ${6:+--adapter "$6"} ;;
  eval)
    uv run python dialogue_pilot.py "$2" ../datasets/samsum_raw "../results/dialogue_curves/$3" \
      --rows test --tasks "${3%%_*}" --name "$4" ${5:+--adapter "$5"} ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
