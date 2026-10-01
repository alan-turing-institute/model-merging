#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Usage, from evaluate/:
#   sbatch evaluate.sbatch <task> <model> [--adapter <dir>] [inspect eval args...]
#   e.g. sbatch evaluate.sbatch xsum google/gemma-3-4b-it -T prompt_variant=short
#        sbatch evaluate.sbatch xsum ../models/gemma3-xsum-self-dist-short \
#            -T prompt_variant=short --limit 50
#        sbatch evaluate.sbatch crime google/gemma-3-4b-it \
#            --adapter ../models/gemma3-crime-full-lora
#
# task is crime, xsum or billsum (runs <task>_task.py). model is a Hub id or
# a full model saved on disk (config.json + model.safetensors); --adapter puts a
# PEFT/LoRA adapter on top of it via the hf-peft provider. Everything after that
# goes to `inspect eval` unchanged. xsum needs -T prompt_variant=short|long|teacher,
# matching the prompt the model was trained on, so it is required here.
#
# The .eval log is written to logs/; read it with `uv run inspect view`.
set -euo pipefail
USAGE="usage: sbatch evaluate.sbatch <crime|xsum|billsum> <model> [--adapter <dir>] [inspect eval args...]"
TASK=${1:?$USAGE}
MODEL=${2:?$USAGE}
shift 2

ADAPTER=
if [[ ${1:-} == --adapter ]]; then
  ADAPTER=${2:?$USAGE}
  shift 2
fi

# Submit from evaluate/: the task files import semantic_metrics from here, and
# relative model paths resolve against it.
cd "$SLURM_SUBMIT_DIR"
TASK_FILE=${TASK}_task.py
[[ -f $TASK_FILE ]] || { echo "no $TASK_FILE in $PWD - submit from evaluate/" >&2; exit 1; }

# Both summarisation tasks build their prompt from the article/bill rather than
# reading a pre-rendered one, so the variant has to be named - a student trained
# on the short prompt and scored on the long one is being measured on a task it
# never saw.
if [[ $TASK == xsum || $TASK == billsum ]] && [[ " $* " != *" prompt_variant="* ]]; then
  echo "$TASK needs -T prompt_variant=... (the prompt the model was trained on)" >&2
  exit 1
fi

# A directory on disk is loaded through model_path; the name after hf/ is then
# only a label in the log.
MODEL_ARGS=()
if [[ -d $MODEL ]]; then
  MODEL_ARGS+=(-M "model_path=$MODEL")
  MODEL_NAME=$(basename "$MODEL")
else
  MODEL_NAME=$MODEL
fi
if [[ -n $ADAPTER ]]; then
  PROVIDER=hf-peft
  MODEL_ARGS+=(-M "adapter_path=$ADAPTER")
else
  PROVIDER=hf
fi

# Required in every job script on Isambard-AI. Without it the cu130 torch build
# cannot see the GPU and Inspect quietly runs the model on the CPU.
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Hugging Face downloads (base models, the semantic scorer's encoder) live in
# the project area, not the 100 GB home quota. See the main README.
#
# HF_HUB_CACHE, NOT HF_HOME. HF_HOME relocates the auth token as well as the
# cache, which silently logs you out: gemma-3-4b-it is a gated repo, so the next
# fetch 401s instead of failing closed, and on an offline compute node it simply
# cannot find the weights. The token stays at ~/.cache/huggingface/token where
# `hf auth login` put it. Checked on this machine: the models actually live in
# $PROJECTDIR/$USER/mm/hf-cache (39 GB, gemma plus the scorer's encoders), and
# $PROJECTDIR/$USER/hf_cache holds datasets only.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECTDIR/$USER/mm/hf-cache}

uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is the CUDA compat lib on LD_LIBRARY_PATH?'"

# Greedy decoding and bf16 weights for every model, whatever its config.json
# says: Inspect's hf provider defaults do_sample to true, and a checkpoint whose
# config says float32 would otherwise load in fp32 next to a bf16 base model.
uv run inspect eval "$TASK_FILE" --model "$PROVIDER/$MODEL_NAME" "${MODEL_ARGS[@]}" \
  -M batch_size=8 -M do_sample=false -M dtype=bfloat16 \
  --log-dir logs "$@"
