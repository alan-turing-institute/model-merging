#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
# The two GPU stages of the self-distilled LoRA, around main's own code:
#   sbatch --job-name=kd-teacher kd_screen.sbatch teacher
#   sbatch --job-name=kd-train   kd_screen.sbatch train [config.yaml]
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
    # Optional second argument: the config to train (default: the 4-bit one).
    CONFIG=${2:-self-distillation_xsum_lora.yaml}
    OUT=$(awk '/^output_dir:/ {print $2}' "$CONFIG")
    mkdir -p "$OUT"
    uv run axolotl preprocess "$CONFIG"
    uv run axolotl train      "$CONFIG"
    ;;
  check)
    # Step-0 check on the eval rows axolotl prepared: bf16, 8-bit and 4-bit.
    # Axolotl never expanded the ${MM_PREPARED:-...} in the config, so its
    # prepared data is in a directory literally named that.
    PREP=${2:-}
    [[ -n $PREP ]] || PREP='${MM_PREPARED:-last_run_prepared}'
    [[ -d $PREP ]] || { echo "no prepared data at $PREP" >&2; exit 1; }
    for mode in "" --load-in-8bit --load-in-4bit; do
      uv run python kd_step0_check.py "$PREP" $mode
    done
    ;;
  gradcheck)
    PREP=${2:-}
    [[ -n $PREP ]] || PREP='${MM_PREPARED:-last_run_prepared}'
    uv run python kd_grad_check.py "$PREP"
    ;;
  probe)
    # Two optimiser steps of the bf16 config with one setting changed at a time,
    # to find which change gives a finite first gradient. Reuses the prepared
    # data; nothing is evaluated or saved.
    for variant in nopad nockpt fp32; do
      PROBE_CFG=$(mktemp --suffix=.yaml -p .)
      uv run python - self-distillation_xsum_lora_bf16.yaml "$PROBE_CFG" "$variant" <<'EOF'
import sys, yaml
src, dst, variant = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg.update(max_steps=2, logging_steps=1, eval_strategy="no", save_strategy="no",
           output_dir=f"../models/kd-probe-{variant}")
cfg.pop("evals_per_epoch", None)
if variant == "nopad":
    cfg.update(micro_batch_size=1, gradient_accumulation_steps=16)
elif variant == "nockpt":
    cfg["gradient_checkpointing"] = False
elif variant == "fp32":
    cfg.update(bf16=False, fp16=False)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF
      echo "=== probe $variant ==="
      uv run axolotl train "$PROBE_CFG" 2>&1 | grep -oE "\{.loss.: [^{}]*\}" | grep -oE "(loss|grad_norm).: .?[0-9a-z.e+-]+" | paste - -
      rm -f "$PROBE_CFG"
    done
    ;;
  nancheck)
    PREP=${2:-}
    [[ -n $PREP ]] || PREP='${MM_PREPARED:-last_run_prepared}'
    uv run python kd_lora_nan_check.py "$PREP"
    ;;
  trainloop)
    # bf16 LoRA trained outside axolotl (kd_lora_train.py), on axolotl's own
    # prepared tensors, because axolotl's KD trainer NaNs on this configuration.
    # Optional: a student model and output directory (default: the 4B run).
    PREP='${MM_PREPARED:-last_run_prepared}'
    STUDENT=${2:-google/gemma-3-4b-it}
    OUT=${3:-../models/gemma3-xsum_kd_screen_bf16loop-lora}
    mkdir -p "$OUT"
    uv run python kd_lora_train.py "$PREP" "$OUT" --model "$STUDENT"
    ;;
  promptsweep)
    # Teacher-prompt comparison on the XSum validation articles: generate in
    # this (vLLM) environment, score in evaluate's, which has the entity scorer.
    OUT=../results/teacher_prompt_sweep
    export HF_HUB_OFFLINE=1
    uv run python teacher_prompt_sweep.py ../datasets/xsum_screen/full "$OUT"
    ../evaluate/.venv/bin/python ../evaluate/teacher_prompt_score.py "$OUT"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
