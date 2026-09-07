#!/usr/bin/env bash
# Runs the full train -> merge -> evaluate pipeline described in the root README.
#
# Storage policy: local disk is SCRATCH ONLY. Nothing valuable is left on it.
#
#   Azure ML registry  - the 3 LoRA adapters, the full-dataset model, and the
#                        merged model. The experimental output and the two
#                        things the merge is compared against: worth versioning,
#                        and referencable as azureml:<name>:<version>, which
#                        evaluate.py resolves natively.
#   Workspace share    - the results JSONs, under ~/cloudfiles when mounted.
#   Local, transient   - the two half-dataset full models (~8.6GB each). Needed
#                        on a real filesystem because mergekit reads weights
#                        from paths, but they exist only as merge inputs, so
#                        they are never registered and are deleted at the end.
#                        Convert_to_full_model.py regenerates them from the
#                        registered adapters in minutes if ever needed again.
#
# Local disk cannot be avoided during compute - axolotl's output_dir, the
# conversion script and mergekit all take filesystem paths, and the registry is
# not a filesystem. Pointing them at ~/cloudfiles would be worse: it is Azure
# Files over SMB, and mergekit's --lazy-unpickle memory-maps multi-GB
# safetensors, which is slow and flaky over a network share. So instead the
# working tree is cleaned up once everything durable has been uploaded.
#
# Cleanup is ON by default. Set KEEP_LOCAL=1 to retain models/ for debugging.
#
# Registration uses the next FREE version per name, never a hardcoded one: these
# names already carry versions registered by other people, so hardcoding either
# collides with their work or silently skips registration and leaves evaluation
# measuring someone else's artifact. Versions this run creates are recorded in
# .pipeline_versions, which is also what makes re-runs safe after cleanup: stages
# are skipped based on what has been REGISTERED, not on what happens to be on
# local disk, and anything needed again is re-fetched from the registry.
#
# Usage: run from the repo root on a GPU compute instance, with `az` working
# (see preflight below) and `hf auth login` already done.
#   RESULTS_DIR=/some/path ./run_pipeline.sh   # override where results land
#   KEEP_LOCAL=1 ./run_pipeline.sh             # skip the cleanup step
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
    # The share is workspace-wide and holds a directory per person, named after
    # the AAD account (not the local azureuser account). Derive which one is
    # ours from the Azure login rather than guessing - picking the wrong one
    # would write results into a colleague's folder.
    cf_user=$(az account show --query user.name -o tsv 2>/dev/null | cut -d@ -f1)
    if [ -n "$cf_user" ] && [ -d "$cf_users/$cf_user" ]; then
      RESULTS_DIR=$cf_users/$cf_user/model-merging-results
    else
      log "Could not identify your folder under $cf_users (az user: ${cf_user:-unknown})"
      log "Set RESULTS_DIR explicitly to use the share, e.g.:"
      log "  RESULTS_DIR=$cf_users/<you>/model-merging-results ./run_pipeline.sh"
    fi
  fi
fi
RESULTS_DIR=${RESULTS_DIR:-$REPO_ROOT/evaluate/results}
mkdir -p "$RESULTS_DIR"
log "Results will be written to $RESULTS_DIR"
case "$RESULTS_DIR" in
  "$REPO_ROOT"/*) log "WARNING: results are on local disk - they will not survive instance deletion" ;;
esac

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

# Make sure a registered artifact is present locally, downloading if cleanup
# (or a fresh instance) removed it. az ml model download places the artifact in
# a directory named after the model under --download-path; if your CLI version
# nests it differently, adjust here rather than at every call site.
ensure_local() {
  local name=$1 path=$2 v
  [ -d "$path" ] && return 0
  v=$(recorded_version "$name")
  [ -n "$v" ] || die "$path is missing and $name was not registered by this run"
  log "Fetching $name:$v from registry -> $path"
  az ml model download --name "$name" --version "$v" \
    --download-path "$(dirname "$path")" \
    --resource-group "$RG" --workspace-name "$WS" >/dev/null
  [ -d "$path" ] || die "Downloaded $name:$v but nothing landed at $path (check nesting)"
}

# --- 1. data prep ---

if [ ! -d ../datasets/crime_dataset ]; then
  log "Preparing data"
  uv run --project evaluate python prepare_data.py
else
  log "Data already prepared, skipping"
fi

# --- 2. train the three LoRA adapters ---
# Skip on REGISTERED state first, so a cleaned-up working tree doesn't retrain.

train_if_needed() {
  local cfg=$1 name=$2 outdir=$3
  if [ -n "$(recorded_version "$name")" ]; then
    log "$name already registered (version $(recorded_version "$name")), skipping training"
    return
  fi
  # Completion is recorded by a sentinel THIS script writes, not inferred from
  # axolotl's output. The previous check looked for trainer_state.json at the
  # top of output_dir, but axolotl only writes that inside checkpoint-N/, so
  # the condition was never true and every unregistered adapter retrained from
  # scratch - confirmed on the XSum arm, which carried the same check.
  # adapter_model.safetensors on its own is no good either: it is pre-saved
  # before step 0, so it exists for an interrupted run too.
  if [ -f "$outdir/.training_complete" ]; then
    log "Adapter already trained at $outdir, skipping"
    return
  fi
  log "Training $cfg"
  (cd train && uv run axolotl train "$cfg")
  [ -f "$outdir/adapter_model.safetensors" ] || die "Training finished but no adapter at $outdir"
  touch "$outdir/.training_complete"
}
train_if_needed crime_gemma.yaml  gemma3-crime-full-lora   models/gemma3-crime-full-lora
train_if_needed crime_gemma1.yaml gemma3-crime-1-of-2-lora models/gemma3-crime-1-of-2-lora
train_if_needed crime_gemma2.yaml gemma3-crime-2-of-2-lora models/gemma3-crime-2-of-2-lora

# --- 3. register the adapters (the trained output - this must land) ---

register_model gemma3-crime-full-lora   models/gemma3-crime-full-lora
register_model gemma3-crime-1-of-2-lora models/gemma3-crime-1-of-2-lora
register_model gemma3-crime-2-of-2-lora models/gemma3-crime-2-of-2-lora

# --- 4. convert adapters into full models ---
# The full-dataset one is registered below; the two halves stay local scratch.

convert_if_needed() {
  local name=$1 lora=$2 outdir=$3
  if [ -f "$outdir/config.json" ]; then
    log "Full model already converted at $outdir, skipping"
    return
  fi
  ensure_local "$name" "$lora"
  log "Converting $lora -> $outdir"
  uv run --project evaluate python Convert_to_full_model.py "$BASE_MODEL" "$lora" "$outdir"
}
convert_if_needed gemma3-crime-full-lora   models/gemma3-crime-full-lora   models/gemma3-crime-full
convert_if_needed gemma3-crime-1-of-2-lora models/gemma3-crime-1-of-2-lora models/gemma3-crime-1-of-2
convert_if_needed gemma3-crime-2-of-2-lora models/gemma3-crime-2-of-2-lora models/gemma3-crime-2-of-2

# The full-dataset model IS registered: it's the baseline the merge is compared
# against, the README lists it as a maintained artifact, and evaluating it as a
# materialised model (rather than base+adapter) is what makes it a distinct data
# point. The two half-dataset models stay transient - they're only merge inputs.
register_model gemma3-crime-full models/gemma3-crime-full

# --- 5. merge the two half-dataset full models (linear) ---
# merge_linear_config.yaml reads ../models/gemma3-crime-{1,2}-of-2, which the
# convert step just produced locally.

if [ -n "$(recorded_version gemma3-crime-merged-linear-2)" ]; then
  log "Merge already registered, skipping"
elif [ -f models/gemma3-crime-merged-linear-2/config.json ]; then
  log "Merge already done locally, skipping"
else
  log "Merging (linear)"
  (cd merge && uv run mergekit-yaml merge_linear_config.yaml \
     ../models/gemma3-crime-merged-linear-2 --cuda --lazy-unpickle --allow-crimes)
fi

# The merge is a result, not a regenerable intermediate - keep it versioned.
register_model gemma3-crime-merged-linear-2 models/gemma3-crime-merged-linear-2

# --- 6. evaluate ---
# Everything evaluated is referenced from the registry at the version THIS run
# created, so evaluation does not depend on local disk surviving.

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

# base model + the full-dataset adapter, applied at load time
evaluate_if_needed "$BASE_MODEL" "$(ref gemma3-crime-full-lora)" \
  gemma3-crime-full-lora.json
# the same thing materialised into full weights - should match the above, and a
# divergence means merge_and_unload() changed behaviour
evaluate_if_needed "$(ref gemma3-crime-full)" "" \
  gemma3-crime-full.json
# the linear merge of the two half-dataset models - the actual experiment
evaluate_if_needed "$(ref gemma3-crime-merged-linear-2)" "" \
  gemma3-crime-merged-linear-2.json

# --- 7. clean up local scratch ---

if [ "${KEEP_LOCAL:-0}" = "1" ]; then
  log "KEEP_LOCAL=1 - leaving models/ in place ($(du -sh models 2>/dev/null | cut -f1) on disk)"
else
  log "Removing local scratch models/ ($(du -sh models 2>/dev/null | cut -f1)) - everything durable is registered"
  rm -rf models
fi

log "Pipeline complete. Versions registered by this run:"
cat "$VERSIONS_FILE" >&2
log "Results in $RESULTS_DIR"
log "Note: the base model cache in ~/.cache/huggingface (~8.6GB) is left in place;"
log "clear it with 'rm -rf ~/.cache/huggingface/hub' if you need the space back."
