#!/usr/bin/env bash
# Runs the full train -> merge -> evaluate pipeline described in the root README.
#
# What gets stored where, and why:
#
#   Azure ML registry  - the LoRA adapters and the merged model. These are the
#                        actual experimental output: small (~tens of MB), worth
#                        versioning so runs stay comparable, and referencable as
#                        azureml:<name>:<version> which evaluate.py resolves.
#   Local disk only    - the converted full models (~8.6GB each). Purely derived:
#                        Convert_to_full_model.py regenerates them deterministically
#                        from base model + adapter, so registering ~26GB per run
#                        buys nothing. The merge step reads them from here.
#   Workspace share    - the results JSONs, under ~/cloudfiles if it's mounted.
#                        Local disk is per-instance and dies with the instance;
#                        results are cheap to store and painful to lose. Model
#                        weights stay OFF the share - it's Azure Files, and
#                        multi-GB reads there are much slower than local SSD.
#
# Registration uses the next FREE version per name, never a hardcoded one: these
# names already carry versions registered by other people, so hardcoding either
# collides with their work or silently skips registration and leaves evaluation
# measuring someone else's artifact. Versions this run creates are recorded in
# .pipeline_versions and reused on re-runs, so re-running never double-registers.
#
# Safe to re-run: each stage is skipped if its output already exists.
#
# Usage: run from the repo root on a GPU compute instance, with `az` working
# (see preflight below) and `hf auth login` already done. Override where results
# land with RESULTS_DIR=/some/path ./run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$PWD

RG=tire-1
WS=tire-2
BASE_MODEL=google/gemma-3-4b-it
VERSIONS_FILE=$REPO_ROOT/.pipeline_versions

# Progress goes to stderr so stdout stays clean for captured values.
log() { echo "=== $* ===" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

# --- where results go: workspace share if mounted, else local ---

if [ -z "${RESULTS_DIR:-}" ]; then
  cf_users=$HOME/cloudfiles/code/Users
  if [ -d "$cf_users" ]; then
    # The share is named after the AAD user, not the local azureuser account.
    cf_user_dir=$(find "$cf_users" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1)
    if [ -n "$cf_user_dir" ]; then
      RESULTS_DIR=$cf_user_dir/model-merging-results
    fi
  fi
fi
RESULTS_DIR=${RESULTS_DIR:-$REPO_ROOT/evaluate/results}
mkdir -p "$RESULTS_DIR"
log "Results will be written to $RESULTS_DIR"

# --- preflight: fail fast, before hours of training ---

for proj in train merge evaluate; do
  if [ ! -d "$proj/.venv" ]; then
    log "Syncing $proj env"
    (cd "$proj" && uv sync)
  fi
done

uv run --project evaluate hf auth whoami >/dev/null 2>&1 \
  || die "Not logged into Hugging Face. Run: uv run --project evaluate hf auth login"

if ! az ml model list --resource-group "$RG" --workspace-name "$WS" >/dev/null 2>&1; then
  echo "ERROR: 'az ml' is not working here - extension missing, or not logged in." >&2
  echo "On an Azure ML compute instance the system extension dir isn't writable, so:" >&2
  echo "  export AZURE_EXTENSION_DIR=\$HOME/.azure/cliextensions" >&2
  echo "  az extension add -n ml -y" >&2
  echo "  az login --use-device-code" >&2
  exit 1
fi

# --- version bookkeeping ---

# Highest version currently registered for a model name, or 0 if none.
highest_version() {
  local name=$1 max
  max=$(az ml model list --name "$name" --resource-group "$RG" --workspace-name "$WS" \
        --query "[].version" -o tsv 2>/dev/null | sort -n | tail -1)
  echo "${max:-0}"
}

# Version this pipeline run registered for a name, if any.
recorded_version() {
  [ -f "$VERSIONS_FILE" ] || return 0
  grep "^$1=" "$VERSIONS_FILE" 2>/dev/null | tail -1 | cut -d= -f2
}

# Register $path under $name at the next free version. Idempotent across re-runs.
register_model() {
  local name=$1 path=$2 existing next
  existing=$(recorded_version "$name")
  if [ -n "$existing" ]; then
    log "$name already registered by this run as version $existing, skipping"
    return
  fi
  [ -d "$path" ] || die "Nothing to register at $path"
  next=$(( $(highest_version "$name") + 1 ))
  log "Registering $name as version $next (from $path)"
  az ml model create --name "$name" --version "$next" --type custom_model \
    --path "$path" --resource-group "$RG" --workspace-name "$WS" >/dev/null
  echo "$name=$next" >> "$VERSIONS_FILE"
}

# The azureml: reference for something this run registered.
ref() {
  local name=$1 v
  v=$(recorded_version "$name")
  [ -n "$v" ] || die "No version recorded for $name - registration must run first"
  echo "azureml:$name:$v"
}

# --- 1. data prep ---

if [ ! -d ../datasets/crime_dataset ]; then
  log "Preparing data"
  uv run --project evaluate python prepare_data.py
else
  log "Data already prepared, skipping"
fi

# --- 2. train the three LoRA adapters ---

train_if_needed() {
  local cfg=$1 outdir=$2
  # trainer_state.json only lands when training actually finishes; the adapter
  # file alone is pre-saved before step 0 and would cause a false skip.
  if [ -f "$outdir/adapter_model.safetensors" ] && [ -f "$outdir/trainer_state.json" ]; then
    log "Adapter already trained at $outdir, skipping"
  else
    log "Training $cfg"
    (cd train && uv run axolotl train "$cfg")
    [ -f "$outdir/adapter_model.safetensors" ] || die "Training finished but no adapter at $outdir"
  fi
}
train_if_needed crime_gemma.yaml  models/gemma3-crime-full-lora
train_if_needed crime_gemma1.yaml models/gemma3-crime-1-of-2-lora
train_if_needed crime_gemma2.yaml models/gemma3-crime-2-of-2-lora

# --- 3. register the adapters (the trained output - this must land) ---

register_model gemma3-crime-full-lora   models/gemma3-crime-full-lora
register_model gemma3-crime-1-of-2-lora models/gemma3-crime-1-of-2-lora
register_model gemma3-crime-2-of-2-lora models/gemma3-crime-2-of-2-lora

# --- 4. convert adapters into full models (local only - derived, ~8.6GB each) ---

convert_if_needed() {
  local lora=$1 outdir=$2
  if [ -f "$outdir/config.json" ]; then
    log "Full model already converted at $outdir, skipping"
  else
    log "Converting $lora -> $outdir"
    uv run --project evaluate python Convert_to_full_model.py "$BASE_MODEL" "$lora" "$outdir"
  fi
}
convert_if_needed models/gemma3-crime-full-lora   models/gemma3-crime-full
convert_if_needed models/gemma3-crime-1-of-2-lora models/gemma3-crime-1-of-2
convert_if_needed models/gemma3-crime-2-of-2-lora models/gemma3-crime-2-of-2

# --- 5. merge the two half-dataset full models (linear) ---
# merge_linear_config.yaml reads ../models/gemma3-crime-{1,2}-of-2, which the
# convert step just produced locally - no registry download needed.

if [ -f models/gemma3-crime-merged-linear-2/config.json ]; then
  log "Merge already done, skipping"
else
  log "Merging (linear)"
  (cd merge && uv run mergekit-yaml merge_linear_config.yaml \
     ../models/gemma3-crime-merged-linear-2 --cuda --lazy-unpickle --allow-crimes)
fi

# The merge is a result, not a regenerable intermediate - keep it versioned.
register_model gemma3-crime-merged-linear-2 models/gemma3-crime-merged-linear-2

# --- 6. evaluate ---
# Registered artifacts are referenced by the version THIS run created; the
# unregistered full model is referenced by its local path.

evaluate_if_needed() {
  local model=$1 adapter=$2 out=$3
  local out_path=$RESULTS_DIR/$out
  if [ -f "$out_path" ]; then
    log "Already evaluated -> $out_path, skipping"
    return
  fi
  log "Evaluating -> $out_path (model=$model adapter=${adapter:-none})"
  if [ -n "$adapter" ]; then
    (cd evaluate && uv run python evaluate.py --model "$model" --adapter "$adapter" --output "$out_path")
  else
    (cd evaluate && uv run python evaluate.py --model "$model" --output "$out_path")
  fi
}

# base model + the full-dataset adapter (registered)
evaluate_if_needed "$BASE_MODEL" "$(ref gemma3-crime-full-lora)" \
  gemma3-crime-full-lora.json
# the full-dataset model (local, unregistered - path is relative to evaluate/)
evaluate_if_needed ../models/gemma3-crime-full "" \
  gemma3-crime-full.json
# the linear merge of the two half-dataset models - the actual experiment
evaluate_if_needed "$(ref gemma3-crime-merged-linear-2)" "" \
  gemma3-crime-merged-linear-2.json

log "Pipeline complete. Versions registered by this run:"
cat "$VERSIONS_FILE" >&2
log "Results in $RESULTS_DIR"
