#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
# The two GPU stages of the self-distilled LoRA, around main's own code:
#   sbatch --job-name=kd-teacher kd_screen.sbatch teacher
#   sbatch --job-name=kd-train   kd_screen.sbatch train
# `teacher` runs generate_teacher_logprobs.py unchanged on the screen's prompts;
# `train` runs axolotl with self-distillation_xsum_lora.yaml. A wrapper of our
# own rather than generate_teacher_logprobs.sbatch / train_xsum.sbatch only for
# the environment: those set HF_HOME to a cache that does not hold the Gemma
# weights, and compute nodes cannot download them.
set -euo pipefail
STAGE=${1:?usage: kd_screen.sbatch <teacher|train>}
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
export UV_PROJECT_ENVIRONMENT=${SELF_DISTILL_VENV:-$PROJECTDIR/$USER/mm/venvs/self-distill}
uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU'"

case $STAGE in
  teacher)
    uv run python generate_teacher_logprobs.py google/gemma-3-4b-it \
      ../datasets/xsum_kd_screen_prompts ../datasets \
      --prefix xsum_kd_screen --student-fields messages_short --max-seq-len 1024
    ;;
  train)
    OUT=../models/gemma3-xsum_kd_screen-lora
    mkdir -p "$OUT"
    export MM_PREPARED="$OUT/prepared"
    uv run axolotl preprocess self-distillation_xsum_lora.yaml
    uv run axolotl train      self-distillation_xsum_lora.yaml
    ;;
  check)
    # Step-0 check on the eval rows axolotl prepared, at both weight formats.
    # Axolotl never expanded the ${MM_PREPARED:-...} in the config, so its
    # prepared data is in a directory literally named that.
    PREP=${2:-}
    [[ -n $PREP ]] || PREP='${MM_PREPARED:-last_run_prepared}'
    [[ -d $PREP ]] || { echo "no prepared data at $PREP" >&2; exit 1; }
    uv run python kd_step0_check.py "$PREP"
    uv run python kd_step0_check.py "$PREP" --load-in-4bit
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
