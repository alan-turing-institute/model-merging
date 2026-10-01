#!/bin/bash
# Run an evaluation task on the Azure ML compute instance.
#
# Usage, from anywhere:
#   Azure/evaluate.sh <task> <model> [--adapter <ref>] [inspect eval args...]
#   e.g. Azure/evaluate.sh crime google/gemma-3-4b-it \
#            --adapter azureml:gemma3-crime-full-lora:1
#        Azure/evaluate.sh xsum azureml:gemma3-xsum-self-dist-short:1 \
#            -T prompt_variant=short --limit 50
#
# The Azure counterpart of evaluate.sbatch: the same `inspect eval` command,
# plus resolving Azure ML model-registry references. task is crime or xsum
# (runs <task>_task.py). model and --adapter each take a local path, a Hugging
# Face hub id, or azureml:<name>:<version>; a registry reference is downloaded
# into evaluate/.azureml_models/ (skipped if already there). xsum needs
# -T prompt_variant=short|long|teacher, matching the prompt the model was
# trained on.
#
# Registry settings, overridable from the environment:
#   AZUREML_RG (default tire-1), AZUREML_WS (default tire-2),
#   AZUREML_SUBSCRIPTION (default: az cli's active subscription).
#
# The .eval log is written to evaluate/logs/; read it with `uv run inspect view`.
set -euo pipefail
USAGE="usage: Azure/evaluate.sh <crime|xsum> <model> [--adapter <ref>] [inspect eval args...]"
TASK=${1:?$USAGE}
MODEL=${2:?$USAGE}
shift 2

ADAPTER=
if [[ ${1:-} == --adapter ]]; then
  ADAPTER=${2:?$USAGE}
  shift 2
fi

RESOURCE_GROUP=${AZUREML_RG:-tire-1}
WORKSPACE=${AZUREML_WS:-tire-2}
SUBSCRIPTION=${AZUREML_SUBSCRIPTION:-}

# Local paths are relative to where this was run from, so pin them down before
# moving to evaluate/ (the task files import semantic_metrics from there).
[[ -e $MODEL ]] && MODEL=$(realpath "$MODEL")
[[ -n $ADAPTER && -e $ADAPTER ]] && ADAPTER=$(realpath "$ADAPTER")
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TASK_FILE=${TASK}_task.py
[[ -f $TASK_FILE ]] || { echo "no $TASK_FILE in $PWD" >&2; exit 1; }

if [[ $TASK == xsum && " $* " != *" prompt_variant="* ]]; then
  echo "xsum needs -T prompt_variant=short|long|teacher (the prompt the model was trained on)" >&2
  exit 1
fi

CACHE_DIR=.azureml_models

# Print the local path for a model reference, downloading an azureml:<name>:<version>
# reference from the registry first. Anything else is printed unchanged.
resolve() {
  local ref=$1
  if [[ $ref != azureml:* ]]; then
    echo "$ref"
    return
  fi
  local name version
  IFS=: read -r _ name version <<<"$ref"
  [[ -n $name && -n $version ]] || { echo "bad reference $ref (want azureml:<name>:<version>)" >&2; exit 1; }
  local download_dir=$CACHE_DIR/$name-$version
  local local_path=$download_dir/$name
  if [[ ! -e $local_path ]]; then
    echo "Downloading $ref to $local_path ..." >&2
    az ml model download --name "$name" --version "$version" \
      --download-path "$download_dir" \
      --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE" \
      ${SUBSCRIPTION:+--subscription "$SUBSCRIPTION"} >&2
  fi
  echo "$local_path"
}

MODEL_PATH=$(resolve "$MODEL")
ADAPTER_PATH=
[[ -n $ADAPTER ]] && ADAPTER_PATH=$(resolve "$ADAPTER")

# A directory on disk is loaded through model_path; the name after hf/ is then
# only a label in the log, so a registry model is labelled with its reference.
MODEL_ARGS=()
if [[ -d $MODEL_PATH ]]; then
  MODEL_ARGS+=(-M "model_path=$MODEL_PATH")
  MODEL_NAME=$(basename "$MODEL_PATH")
else
  MODEL_NAME=$MODEL_PATH
fi
if [[ -n $ADAPTER_PATH ]]; then
  PROVIDER=hf-peft
  MODEL_ARGS+=(-M "adapter_path=$ADAPTER_PATH")
else
  PROVIDER=hf
fi

# Record the registry references in the log, so a result evaluated from a
# local copy still says which registered version produced it.
METADATA=()
[[ $MODEL == azureml:* ]] && METADATA+=(--metadata "model_ref=$MODEL")
[[ $ADAPTER == azureml:* ]] && METADATA+=(--metadata "adapter_ref=$ADAPTER")

uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: torch cannot see a CUDA device'"

# Greedy decoding and bf16 weights for every model, as in evaluate.sbatch.
uv run inspect eval "$TASK_FILE" --model "$PROVIDER/$MODEL_NAME" "${MODEL_ARGS[@]}" \
  -M batch_size=8 -M do_sample=false -M dtype=bfloat16 \
  "${METADATA[@]}" --log-dir logs "$@"
