#!/usr/bin/env bash
# Runs the full train -> merge -> evaluate pipeline described in the root README.
#
# Merges LoRA adapters DIRECTLY, via mergekit's <base_model>+<adapter> syntax.
# It does not convert adapters into full models first: the root README states
# you should not need to, and doing so cost ~8.6GB per adapter (~26GB for this
# experiment) purely to hold intermediates mergekit can derive itself.
# Convert_to_full_model.py remains available for other purposes; the pipeline
# just doesn't depend on it.
#
# What gets stored where:
#
#   Azure ML registry  - the LoRA adapters and every merged model. The
#                        experimental output: worth versioning, and
#                        referencable as azureml:<name>:<version>, which
#                        evaluate.py resolves natively.
#   Workspace share    - the results JSONs, under ~/cloudfiles when mounted.
#   Local, transient   - mergekit's --lora-merge-cache (one materialised
#                        base+adapter per adapter, ~8.6GB each) and one merge
#                        output at a time. Both are deleted at the end.
#
# Local disk cannot be avoided during compute - axolotl's output_dir and
# mergekit both take filesystem paths, and the registry is not a filesystem.
# Pointing them at ~/cloudfiles would be worse: it is Azure Files over SMB, and
# --lazy-unpickle memory-maps multi-GB safetensors, which is slow and flaky
# over a network share. So the working tree is cleaned once everything durable
# has been uploaded.
#
# Registration uses the next FREE version per name, never a hardcoded one:
# these names already carry versions registered by other people, so hardcoding
# either collides with their work or silently skips registration and leaves
# evaluation measuring someone else's artifact. Versions this run creates are
# recorded in .pipeline_versions, which is also what makes re-runs safe after
# cleanup: stages are skipped on what has been REGISTERED, not on what happens
# to be on local disk, and anything needed again is re-fetched.
#
# Usage, from the repo root:
#   ./run_pipeline.sh
#   METHODS="linear ties dare_ties" ./run_pipeline.sh   # which merges to run
#   RESULTS_DIR=/some/path ./run_pipeline.sh            # where results land
#   KEEP_LOCAL=1 ./run_pipeline.sh                      # skip cleanup
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$PWD

RG=tire-1
WS=tire-2
BASE_MODEL=google/gemma-3-4b-it
VERSIONS_FILE=$REPO_ROOT/.pipeline_versions
CACHE=$REPO_ROOT/models/.lora_merge_cache

# `linear` is the baseline the project started from; `ties` is the only method
# so far to significantly beat both half-data models, so both run by default.
# merge/ also holds task_arithmetic, dare_ties, model_stock, slerp and
# arcee_fusion - add them via METHODS. Note model_stock needs the base model
# plus 3+ others to estimate its angle, so it degenerates on a 2-model merge.
METHODS=${METHODS:-"linear ties"}

# Adapter name -> training config, in training order.
ADAPTERS=(
  "gemma3-crime-full-lora:crime_gemma.yaml"
  "gemma3-crime-1-of-2-lora:crime_gemma1.yaml"
  "gemma3-crime-2-of-2-lora:crime_gemma2.yaml"
)

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

for method in $METHODS; do
  [ -f "merge/merge_${method}_config.yaml" ] \
    || die "No config at merge/merge_${method}_config.yaml (METHODS=\"$METHODS\")"
done

# The adapter cache holds one materialised base+adapter per adapter referenced,
# and each merge writes another full model, so check before starting rather
# than dying part-way through a merge.
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
avail=$(free_gb "$REPO_ROOT")
if [ -n "$avail" ]; then
  log "Free space: ${avail}GB"
  if [ "$avail" -lt 30 ]; then
    die "Need ~30GB free (~17GB adapter cache + ~8.6GB per merge output).
Free some first, e.g.:
  rm -rf $REPO_ROOT/models
  rm -rf ~/.cache/huggingface/hub   # forces an 8.6GB base-model re-download"
  fi
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

# Registry name for a merge method. Config files use underscores
# (merge_dare_ties_config.yaml) but the registered artifacts and the committed
# results/ files use hyphens, and `linear` is historically registered as
# gemma3-crime-merged-linear-2 (the "-2" meaning a 2-way merge). Reusing those
# exact names keeps this run in the same version history rather than starting
# a parallel set of near-identical names.
merged_name() {
  case $1 in
    linear) echo "gemma3-crime-merged-linear-2" ;;
    *)      echo "gemma3-crime-merged-${1//_/-}" ;;
  esac
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

# --- 2. train the LoRA adapters ---
# Skip on REGISTERED state first, so a cleaned-up working tree doesn't retrain.

for entry in "${ADAPTERS[@]}"; do
  name=${entry%%:*}
  cfg=${entry#*:}
  outdir=models/$name

  if [ -n "$(recorded_version "$name")" ]; then
    log "$name already registered (version $(recorded_version "$name")), skipping training"
    continue
  fi
  # trainer_state.json only lands when training actually finishes; the adapter
  # file alone is pre-saved before step 0 and would cause a false skip.
  if [ -f "$outdir/adapter_model.safetensors" ] && [ -f "$outdir/trainer_state.json" ]; then
    log "$name already trained at $outdir, skipping"
    continue
  fi
  log "Training $cfg -> $outdir"
  (cd train && uv run axolotl train "$cfg")
  [ -f "$outdir/adapter_model.safetensors" ] || die "Training finished but no adapter at $outdir"
done

# --- 3. register the adapters (the trained output - this must land) ---

for entry in "${ADAPTERS[@]}"; do
  name=${entry%%:*}
  register_model "$name" "models/$name"
done

# --- 4. merge the two half-dataset adapters, once per method ---
# Every config in merge/ references <base_model>+<adapter>, so mergekit has to
# materialise each combination into --lora-merge-cache first. That cache is the
# same for every method, so the first merge builds it and the rest reuse it.

for name in gemma3-crime-1-of-2-lora gemma3-crime-2-of-2-lora; do
  ensure_local "$name" "models/$name"
done

for method in $METHODS; do
  merged=$(merged_name "$method")
  out=models/$merged

  if [ -n "$(recorded_version "$merged")" ]; then
    log "$merged already registered, skipping merge"
    continue
  fi
  if [ -f "$out/config.json" ]; then
    log "$merged already merged locally, skipping"
  else
    log "Merging: $method"
    # A method can legitimately fail on this input (model_stock needs 3+
    # models, slerp exactly 2). Don't abort the whole run for one method.
    if ! (cd merge && uv run mergekit-yaml "merge_${method}_config.yaml" "../$out" \
          --cuda --lazy-unpickle --allow-crimes --lora-merge-cache "$CACHE"); then
      log "$method FAILED to merge - continuing with the remaining methods"
      rm -rf "$out"
      continue
    fi
  fi
  register_model "$merged" "$out"

  # Free the merge output now rather than accumulating ~8.6GB per method.
  # Evaluation below re-downloads it via its azureml: reference, which is a
  # deliberate round trip: it means each result JSON records
  # azureml:<name>:<version> rather than a local path, so a result can be
  # traced back to the exact artifact that produced it. Provenance is worth
  # more here than the transfer, and keeping every output would otherwise push
  # peak disk to ~17GB of cache plus ~8.6GB per method.
  if [ "${KEEP_LOCAL:-0}" != "1" ]; then
    log "Removing local $out (registered as $(ref "$merged"))"
    rm -rf "$out"
  fi
done

# --- 5. evaluate ---
# Everything is referenced from the registry at the version THIS run created,
# so evaluation does not depend on local disk surviving.

evaluate_if_needed() {
  local model=$1 adapter=$2 out=$3
  local out_path=$RESULTS_DIR/$out
  if [ -f "$out_path" ]; then
    log "Already evaluated -> $out_path, skipping"
    return
  fi
  log "Evaluating -> $out_path (model=$model adapter=${adapter:-none})"
  # evaluate.py prints no progress during its per-example generation loop, so
  # expect several silent minutes per model over the ~1077-example test set.
  if [ -n "$adapter" ]; then
    (cd evaluate && uv run python evaluate.py --model "$model" --adapter "$adapter" --output "$out_path")
  else
    (cd evaluate && uv run python evaluate.py --model "$model" --output "$out_path")
  fi
}

# Each adapter on top of the base model: the full-data baseline, and the two
# half-data models the merges are trying to beat.
for entry in "${ADAPTERS[@]}"; do
  name=${entry%%:*}
  evaluate_if_needed "$BASE_MODEL" "$(ref "$name")" "$name.json"
done

# Each merged model that registered successfully.
for method in $METHODS; do
  merged=$(merged_name "$method")
  if [ -n "$(recorded_version "$merged")" ]; then
    evaluate_if_needed "$(ref "$merged")" "" "$merged.json"
  else
    log "$merged was not registered (merge failed?), nothing to evaluate"
  fi
done

# --- 6. clean up local scratch ---

if [ "${KEEP_LOCAL:-0}" = "1" ]; then
  log "KEEP_LOCAL=1 - leaving scratch in place ($(du -sh models evaluate/.azureml_models 2>/dev/null | tail -1 | cut -f1))"
else
  log "Removing local scratch - everything durable is registered"
  # models/ holds the adapters, the shared adapter cache and any retained merge
  # output; evaluate/.azureml_models holds what evaluate.py pulled back from
  # the registry. Both are reconstructible from the registry.
  for scratch in models evaluate/.azureml_models; do
    [ -e "$scratch" ] || continue
    log "  $scratch ($(du -sh "$scratch" 2>/dev/null | cut -f1))"
    rm -rf "$scratch"
  done
fi

log "Pipeline complete. Versions registered by this run:"
cat "$VERSIONS_FILE" >&2
log "Results in $RESULTS_DIR"
log "Compare them with:"
log "  uv run --project evaluate python experiments/mcnemar.py $RESULTS_DIR"
log "Note: the base model cache in ~/.cache/huggingface (~8.6GB) is left in place;"
log "clear it with 'rm -rf ~/.cache/huggingface/hub' if you need the space back."
