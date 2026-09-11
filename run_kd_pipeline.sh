#!/usr/bin/env bash
# Self-distillation onto a merged model. TASK=xsum (default) or TASK=crime.
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
#   teacher 1  = <task>-1-of-2-lora  over  <task>_dataset1/train
#   teacher 2  = <task>-2-of-2-lora  over  <task>_dataset2/train
#   student    = the linear merge of those two  + LoRA
#
# On XSum the shortfall to look for is stylistic before it is numeric: output
# length and sentence count drifting back toward base behaviour. The evaluator
# reports both alongside ROUGE.
#
# Axolotl's KD is OFFLINE, so step 4 is the expensive part: a forward pass of
# each teacher over its half, storing top-k logprobs per assistant token.
#
# STATUS: exploratory. PREREGISTRATION.md licenses three runs and this is not
# one of them; see its Deviations section. It generates a hypothesis, it does
# not test one.
#
# Usage, from the repo root on a GPU compute instance, under screen:
#   ./run_kd_pipeline.sh                                 # XSum
#   TASK=crime ./run_kd_pipeline.sh
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

TASK=${TASK:-xsum}
case "$TASK" in
  xsum)
    DATA_PREP=prepare_xsum_data.py
    DATASET=xsum_dataset
    HALF1_NAME=gemma3-xsum-1-of-2-lora
    HALF2_NAME=gemma3-xsum-2-of-2-lora
    MERGE_NAME=gemma3-xsum-merged-linear
    FULL_NAME=gemma3-xsum-full-lora
    KD_CONFIG=xsum_gemma_kd.yaml
    KD_STUDENT=gemma3-xsum-kd-from-merge
    # What the half-experts were trained for - used to verify their scale.
    DEFAULT_EPOCHS=2
    INSPECT_EVAL=run_inspect_xsum.py
    LEGACY_EVAL=evaluate_summarisation.py
    LEGACY_TAKES_BATCH_SIZE=1
    ;;
  crime)
    DATA_PREP=prepare_data.py
    DATASET=crime_dataset
    HALF1_NAME=gemma3-crime-1-of-2-lora
    HALF2_NAME=gemma3-crime-2-of-2-lora
    MERGE_NAME=gemma3-crime-merged-linear-2
    FULL_NAME=gemma3-crime-full-lora
    KD_CONFIG=crime_gemma_kd.yaml
    KD_STUDENT=gemma3-crime-kd-from-merge
    DEFAULT_EPOCHS=3
    INSPECT_EVAL=run_inspect_crime.py
    LEGACY_EVAL=evaluate.py
    LEGACY_TAKES_BATCH_SIZE=0
    ;;
  *) echo "ERROR: TASK must be xsum or crime, got '$TASK'" >&2; exit 1 ;;
esac

TOP_K=${TOP_K:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
# Which classification evaluator. `inspect` runs crime_task.py through Inspect
# AI - batched, with a browsable .eval log per run. `legacy` is evaluate.py's
# one-example-at-a-time loop. Both write the SAME results JSON, so mcnemar.py
# reads either.
# `inspect` runs the task through Inspect AI - batched, with a browsable .eval
# log per run. `legacy` uses the hand-rolled script. Both write the same results
# JSON, so the analysis tools read either.
KD_EVAL=${KD_EVAL:-inspect}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-$DEFAULT_EPOCHS}
EVAL_LIMIT=${EVAL_LIMIT:-}

source "$REPO_ROOT/pipeline_lib.sh"

pipeline_resolve_results_dir "model-merging-results/kd-$TASK"
pipeline_preflight train merge evaluate

mkdir -p models

# --- 1. data ---

if [ ! -d "../datasets/$DATASET" ]; then
  log "Preparing $TASK data"
  uv run --project evaluate python "$DATA_PREP"
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

# Each artifact is either fetched from the registry at an explicit version, or
# taken from a path you already have - a colleague's handover on the workspace
# share, say. Paths win when set, and skip the download entirely.
#   HALF1_PATH=$HOME/cloudfiles/code/Users/jmcinroy/model-merging/models/gemma3-crime-1-of-2-lora
resolve_artifact() {
  local name=$1 version=$2 override=$3
  local path=$REPO_ROOT/models/$name
  if [ -n "$override" ]; then
    echo "$override"
    return
  fi
  fetch_model "$name" "$version" "models/$name" >&2
  echo "$path"
}

# What a result says it came from must match where it actually came from.
artifact_provenance() {
  local name=$1 version=$2 override=$3
  if [ -n "$override" ]; then echo "path:$override"; else echo "azureml:$name:$version"; fi
}
HALF1_PROV=$(artifact_provenance "$HALF1_NAME" "$HALF1_VERSION" "${HALF1_PATH:-}")
HALF2_PROV=$(artifact_provenance "$HALF2_NAME" "$HALF2_VERSION" "${HALF2_PATH:-}")
MERGE_PROV=$(artifact_provenance "$MERGE_NAME" "$MERGE_VERSION" "${MERGE_PATH:-}")
FULL_PROV=$(artifact_provenance "$FULL_NAME"  "$FULL_VERSION"  "${FULL_PATH:-}")

HALF1=$(resolve_artifact "$HALF1_NAME" "$HALF1_VERSION" "${HALF1_PATH:-}")
HALF2=$(resolve_artifact "$HALF2_NAME" "$HALF2_VERSION" "${HALF2_PATH:-}")
MERGE=$(resolve_artifact "$MERGE_NAME" "$MERGE_VERSION" "${MERGE_PATH:-}")
FULL=$(resolve_artifact "$FULL_NAME"  "$FULL_VERSION"  "${FULL_PATH:-}")

# The training config names the student by a fixed relative path, so an
# overridden merge has to be reachable there. Stage a directory of symlinks
# rather than one symlink, because the staged copy may need files the source
# lacks: mergekit does not write the image-processor configs a multimodal base
# ships, and axolotl loads a processor for gemma-3 whether or not the task
# involves images. Staging also works against a read-only handover directory.
if [ "$MERGE" != "$REPO_ROOT/models/$MERGE_NAME" ]; then
  uv run --project evaluate python stage_model_dir.py \
    --source "$MERGE" --dest "$REPO_ROOT/models/$MERGE_NAME" --base-model "$BASE_MODEL"
fi

# --- 3. verify the teachers are what we think they are ---
# Before any expensive step. A smoke-scale teacher would produce perfectly
# well-formed logprobs and teach the student nothing worth learning, and the
# only symptom would be a disappointing number hours later.

log "Verifying teacher training scale"
uv run --project evaluate python check_adapter_scale.py \
  --adapter "$HALF1" --dataset "../datasets/${DATASET}1/train" --epochs "$TRAIN_EPOCHS" \
  || die "Teacher 1 does not match the dataset it is paired with - set HALF1_VERSION"
uv run --project evaluate python check_adapter_scale.py \
  --adapter "$HALF2" --dataset "../datasets/${DATASET}2/train" --epochs "$TRAIN_EPOCHS" \
  || die "Teacher 2 does not match the dataset it is paired with - set HALF2_VERSION"

# --- 4. teacher logprobs, each expert over its own half ---

precompute_if_needed() {
  local adapter=$1 dataset=$2 out=$3
  # A sentinel, not directory existence. An interrupted precompute leaves the
  # directory behind, and skipping on that means training against a partial -
  # or worse, never-alignment-checked - set of teacher distributions. The
  # alignment check inside precompute_logprobs.py is the only thing standing
  # between this pipeline and a silently misaligned distillation, so it must
  # not be skippable by accident.
  # The sentinel also implies the column name the current code writes. A
  # dataset produced before the rename to "logprobs" carries the old column and
  # would fail deep inside axolotl, so it must not be reused.
  if [ -f "$out/.precompute_complete" ]; then
    log "Logprobs already at $out (verified), skipping"
    return
  fi
  if [ -d "$out" ]; then
    log "Discarding unverified logprobs at $out - no completion sentinel"
    rm -rf "$out"
  fi
  log "Teacher logprobs: $adapter over $dataset"
  uv run --project evaluate python precompute_logprobs.py \
    --model "$BASE_MODEL" --adapter "$adapter" \
    --dataset "$dataset" --output "$out" --top-k "$TOP_K"
  # Written only after the script exits 0, which it does only if its
  # token-count and alignment checks both passed.
  touch "$out/.precompute_complete"
}

precompute_if_needed "$HALF1" "../datasets/${DATASET}1/train" "../datasets/${TASK}_kd_half1"
precompute_if_needed "$HALF2" "../datasets/${DATASET}2/train" "../datasets/${TASK}_kd_half2"

if [ ! -d "../datasets/${TASK}_kd_train" ]; then
  log "Combining the two teachers' halves"
  uv run --project evaluate python concat_datasets.py \
    "../datasets/${TASK}_kd_half1" "../datasets/${TASK}_kd_half2" \
    --output "../datasets/${TASK}_kd_train"
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
  (cd train && uv run axolotl train "$KD_CONFIG")
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
  if [ "$KD_EVAL" = "inspect" ]; then
    args+=(--batch-size "$EVAL_BATCH_SIZE" --log-dir "$RESULTS_DIR/inspect-logs")
    (cd evaluate && uv run python "$INSPECT_EVAL" "${args[@]}")
  else
    if [ "$LEGACY_TAKES_BATCH_SIZE" = "1" ]; then
      args+=(--batch-size "$EVAL_BATCH_SIZE")
    fi
    (cd evaluate && uv run python "$LEGACY_EVAL" "${args[@]}")
  fi
}

evaluate_if_needed kd-from-merge.json "$MERGE" "$REPO_ROOT/models/$KD_STUDENT" "$(ref "$KD_STUDENT")"
evaluate_if_needed merge.json         "$MERGE" "" "$MERGE_PROV"
evaluate_if_needed half1.json         "$BASE_MODEL" "$HALF1" "$HALF1_PROV"
evaluate_if_needed half2.json         "$BASE_MODEL" "$HALF2" "$HALF2_PROV"
evaluate_if_needed full.json          "$BASE_MODEL" "$FULL"  "$FULL_PROV"

log "Done. Results in $RESULTS_DIR"
log "Recovery fraction R = (kd - best_half) / (full - best_half); compare with:"
if [ "$TASK" = "xsum" ]; then
  log "  uv run --project evaluate python experiments/bootstrap_rouge.py $RESULTS_DIR"
else
  log "  uv run --project evaluate python experiments/mcnemar.py $RESULTS_DIR"
fi
log "Per-sample view: inspect view --log-dir $RESULTS_DIR/inspect-logs"
