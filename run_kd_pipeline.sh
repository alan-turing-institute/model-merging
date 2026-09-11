#!/usr/bin/env bash
# Self-distillation onto a merged model, crime-classification arm.
#
# THE QUESTION. The pilot's linear merge of two half-data experts lost to the
# better of its own two inputs. Merging halves both task vectors and pulls the
# model back toward base behaviour. Does a cheap distillation pass, over data
# the experts have already seen, recover what merging destroyed?
#
# THE DESIGN. Each half-expert teaches its OWN half; the student is initialised
# from the merge. Nothing here requires a machine to have seen all the data -
# the logprob pass shards exactly the way training did - so it stays inside the
# "poor man's parallelism" framing the project is testing. A distillation that
# needed the full dataset in one place would answer a different question.
#
#   teacher 1  = gemma3-crime-1-of-2-lora  over  crime_dataset1/train
#   teacher 2  = gemma3-crime-2-of-2-lora  over  crime_dataset2/train
#   student    = gemma3-crime-merged-linear-2  + LoRA
#
# Axolotl's KD is OFFLINE, so step 4 is the expensive part: a forward pass of
# each teacher over its half, storing top-k logprobs per assistant token.
#
# STATUS: exploratory. PREREGISTRATION.md licenses three runs and this is not
# one of them; see its Deviations section. It generates a hypothesis, it does
# not test one.
#
# Usage, from the repo root on a GPU compute instance, under screen:
#   ./run_kd_pipeline.sh
#   EVAL_LIMIT=100 ./run_kd_pipeline.sh                  # quick shakedown
#   HALF1_VERSION=2 HALF2_VERSION=2 ./run_kd_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$PWD

RG=${AZUREML_RG:-tire-1}
WS=${AZUREML_WS:-tire-2}
BASE_MODEL=google/gemma-3-4b-it
VERSIONS_FILE=$REPO_ROOT/.pipeline_versions_kd

# Explicit, not "latest". These names carry several versions each, and on the
# XSum arm picking by recency silently selected 200-example smoke artifacts.
# Step 3 verifies the choice rather than trusting it.
HALF1_VERSION=${HALF1_VERSION:-3}
HALF2_VERSION=${HALF2_VERSION:-3}
MERGE_VERSION=${MERGE_VERSION:-3}
FULL_VERSION=${FULL_VERSION:-3}

TOP_K=${TOP_K:-64}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-3}     # what the half-experts were trained for
EVAL_LIMIT=${EVAL_LIMIT:-}
KD_STUDENT=gemma3-crime-kd-from-merge

source "$REPO_ROOT/pipeline_lib.sh"

pipeline_resolve_results_dir model-merging-results/kd
pipeline_preflight train merge evaluate

mkdir -p models

# --- 1. data ---

if [ ! -d ../datasets/crime_dataset ]; then
  log "Preparing crime data"
  uv run --project evaluate python prepare_data.py
else
  log "Data already prepared, skipping"
fi

# --- 2. fetch the teachers, the student init, and the comparison models ---

fetch_model() {
  local name=$1 version=$2 path=$3
  if model_dir_has_weights "$path"; then
    log "$name already present at $path"
    return
  fi
  log "Fetching $name:$version"
  az ml model download --name "$name" --version "$version" \
    --download-path models --resource-group "$RG" --workspace-name "$WS" >/dev/null
  flatten_download "$path" || die "Downloaded $name:$version but found no model under $path"
}

HALF1=models/gemma3-crime-1-of-2-lora
HALF2=models/gemma3-crime-2-of-2-lora
MERGE=models/gemma3-crime-merged-linear-2
FULL=models/gemma3-crime-full-lora

fetch_model gemma3-crime-1-of-2-lora     "$HALF1_VERSION" "$HALF1"
fetch_model gemma3-crime-2-of-2-lora     "$HALF2_VERSION" "$HALF2"
fetch_model gemma3-crime-merged-linear-2 "$MERGE_VERSION" "$MERGE"
fetch_model gemma3-crime-full-lora       "$FULL_VERSION"  "$FULL"

# --- 3. verify the teachers are what we think they are ---
# Before any expensive step. A smoke-scale teacher would produce perfectly
# well-formed logprobs and teach the student nothing worth learning, and the
# only symptom would be a disappointing number hours later.

log "Verifying teacher training scale"
uv run --project evaluate python check_adapter_scale.py \
  --adapter "$HALF1" --dataset ../datasets/crime_dataset1/train --epochs "$TRAIN_EPOCHS" \
  || die "Teacher 1 does not match the dataset it is paired with - set HALF1_VERSION"
uv run --project evaluate python check_adapter_scale.py \
  --adapter "$HALF2" --dataset ../datasets/crime_dataset2/train --epochs "$TRAIN_EPOCHS" \
  || die "Teacher 2 does not match the dataset it is paired with - set HALF2_VERSION"

# --- 4. teacher logprobs, each expert over its own half ---

precompute_if_needed() {
  local adapter=$1 dataset=$2 out=$3
  if [ -d "$out" ]; then
    log "Logprobs already at $out, skipping"
    return
  fi
  log "Teacher logprobs: $adapter over $dataset"
  uv run --project evaluate python precompute_logprobs.py \
    --model "$BASE_MODEL" --adapter "$adapter" \
    --dataset "$dataset" --output "$out" --top-k "$TOP_K"
}

precompute_if_needed "$REPO_ROOT/$HALF1" ../datasets/crime_dataset1/train ../datasets/crime_kd_half1
precompute_if_needed "$REPO_ROOT/$HALF2" ../datasets/crime_dataset2/train ../datasets/crime_kd_half2

if [ ! -d ../datasets/crime_kd_train ]; then
  log "Combining the two teachers' halves"
  uv run --project evaluate python concat_datasets.py \
    ../datasets/crime_kd_half1 ../datasets/crime_kd_half2 \
    --output ../datasets/crime_kd_train
else
  log "Combined KD dataset already exists, skipping"
fi

# --- 5. distil ---

if [ -n "$(recorded_version "$KD_STUDENT")" ]; then
  log "$KD_STUDENT already registered, skipping training"
elif [ -f "models/$KD_STUDENT/.training_complete" ]; then
  log "$KD_STUDENT already trained, skipping"
else
  log "Distilling (student init = the linear merge)"
  (cd train && uv run axolotl train crime_gemma_kd.yaml)
  [ -f "models/$KD_STUDENT/adapter_model.safetensors" ] \
    || die "Training finished but no adapter at models/$KD_STUDENT"
  touch "models/$KD_STUDENT/.training_complete"
fi

register_model "$KD_STUDENT" "models/$KD_STUDENT"

# --- 6. evaluate, against everything the student has to beat ---
# The merge is the thing distillation is meant to repair; the better half is
# what a practitioner would otherwise ship; the full model is the target.

evaluate_if_needed() {
  local out=$1 model=$2 adapter=$3 provenance=$4
  local out_path=$RESULTS_DIR/$out
  if [ -f "$out_path" ]; then log "Already evaluated -> $out_path"; return; fi
  log "Evaluating -> $out_path"
  local args=(--model "$model" --output "$out_path")
  if [ -n "$adapter" ]; then args+=(--adapter "$adapter"); fi
  if [ -n "$EVAL_LIMIT" ]; then args+=(--limit "$EVAL_LIMIT"); fi
  if [ -n "$provenance" ]; then args+=(--provenance "$provenance"); fi
  (cd evaluate && uv run python evaluate.py "${args[@]}")
}

evaluate_if_needed kd-from-merge.json "$REPO_ROOT/$MERGE" "$REPO_ROOT/models/$KD_STUDENT" "$(ref "$KD_STUDENT")"
evaluate_if_needed merge.json         "$REPO_ROOT/$MERGE" "" "azureml:gemma3-crime-merged-linear-2:$MERGE_VERSION"
evaluate_if_needed half1.json         "$BASE_MODEL" "$REPO_ROOT/$HALF1" "azureml:gemma3-crime-1-of-2-lora:$HALF1_VERSION"
evaluate_if_needed half2.json         "$BASE_MODEL" "$REPO_ROOT/$HALF2" "azureml:gemma3-crime-2-of-2-lora:$HALF2_VERSION"
evaluate_if_needed full.json          "$BASE_MODEL" "$REPO_ROOT/$FULL"  "azureml:gemma3-crime-full-lora:$FULL_VERSION"

log "Done. Results in $RESULTS_DIR"
log "Recovery fraction R = (kd - best_half) / (full - best_half); compare with:"
log "  uv run --project evaluate python experiments/mcnemar.py $RESULTS_DIR"
