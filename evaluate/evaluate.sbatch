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
# task is crime, xsum, billsum or samsum (runs <task>_task.py). model is a Hub id or
# a full model saved on disk (config.json + model.safetensors); --adapter puts a
# PEFT/LoRA adapter on top of it via the hf-peft provider. Everything after that
# goes to `inspect eval` unchanged. xsum needs -T prompt_variant=short|long|teacher,
# matching the prompt the model was trained on, so it is required here.
#
# The .eval log is written to logs/; read it with `uv run inspect view`.
set -euo pipefail
USAGE="usage: sbatch evaluate.sbatch <crime|xsum|billsum|samsum|clinc150|mt_domain> <model> [--adapter <dir>] [inspect eval args...]"
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

# Every summarisation task builds its prompt from the document rather than
# reading a pre-rendered one, so the variant has to be named - a student trained
# on the short prompt and scored on the long one is being measured on a task it
# never saw. samsum is in this list for the same reason the other two are: its
# default happens to match the training default today, which makes an omission
# silent rather than harmless.
if [[ $TASK == xsum || $TASK == billsum || $TASK == samsum || $TASK == clinc150 ]] && [[ " $* " != *" prompt_variant="* ]]; then
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
# CUDA 13.0 FORWARD-COMPAT LIBS - ONLY WHEN THEY ARE NEWER THAN THE NODE DRIVER.
# Until early October 2026 the nodes ran driver 565.57.01 (CUDA 12.7), and a cu130
# torch build could only see the GPU with the compat libcuda ahead of the system
# one; without it torch fell back to CPU in silence. Isambard then upgraded the
# nodes to 580.173.02, which supports CUDA 13.0 natively - and the compat libcuda
# (580.65.06) became OLDER than the kernel driver. An older userspace libcuda
# against a newer kernel module is CUDA error 803 ("unsupported display driver /
# cuda driver combination"), so the unconditional export that used to be
# required started failing every GPU job. Measured on nid010593:
#   no compat:   torch 2.11.0+cu130, cuda 13.0, is_available() True
#   with compat: torch 2.11.0+cu130, cuda 13.0, is_available() False
# Either way the job runs the same cu130 torch build on the GPU, which is the
# comparability requirement; the version check below keeps that true across a
# driver change in either direction.
CUDA_COMPAT=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat
sys_libcuda=$(ls /usr/lib64/libcuda.so.*.* 2>/dev/null | sed 's/.*libcuda\.so\.//' | sort -V | tail -1)
compat_libcuda=$(ls "$CUDA_COMPAT"/libcuda.so.*.* 2>/dev/null | sed 's/.*libcuda\.so\.//' | sort -V | tail -1)
if [[ -n $compat_libcuda && $compat_libcuda != "$sys_libcuda" \
      && $(printf '%s\n%s\n' "$sys_libcuda" "$compat_libcuda" | sort -V | tail -1) == "$compat_libcuda" ]]; then
  export LD_LIBRARY_PATH=$CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  echo "cuda: driver $sys_libcuda older than compat $compat_libcuda - using compat libs"
else
  echo "cuda: driver ${sys_libcuda:-unknown} not older than compat ${compat_libcuda:-none} - using system libcuda"
fi

# Never re-resolve the environment inside a job. Several array tasks share one
# venv, and concurrent `uv run` syncs race on it; worse, a sync can silently move
# torch to a different build mid-curve. Sync once on a login node instead.
export UV_NO_SYNC=1

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
