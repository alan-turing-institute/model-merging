# Shared helpers for the end-to-end pipeline scripts. Source this, don't run it.
#
# Everything here is task-agnostic: results-directory resolution, preflight
# checks, and the Azure ML version bookkeeping that makes a run resumable.
# The task-specific part - which datasets, which configs, which models get
# registered - lives in the calling script.
#
# The caller must set, before sourcing:
#   REPO_ROOT       absolute path to the repo root
#   RG / WS         Azure resource group and workspace
#   VERSIONS_FILE   where this arm records the versions it registered
#
# run_pipeline.sh (the crime-classification arm) predates this file and still
# carries its own copy of these helpers. Switching it over is a safe follow-up,
# but it is working code mid-experiment so it is deliberately left alone here.

# Progress goes to stderr so stdout stays clean for captured values.
log() { echo "=== $* ===" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

# --- where results go: workspace share if mounted, else local ---

# Sets RESULTS_DIR (respecting an existing value). $1 is the subdirectory name
# to use under the user's folder on the workspace share.
pipeline_resolve_results_dir() {
  local share_subdir=$1 cf_users cf_user
  if [ -z "${RESULTS_DIR:-}" ]; then
    cf_users=$HOME/cloudfiles/code/Users
    if [ -d "$cf_users" ]; then
      # The share is workspace-wide and holds a directory per person, named
      # after the AAD account (not the local azureuser account). Derive which
      # one is ours from the Azure login rather than guessing - picking the
      # wrong one would write results into a colleague's folder.
      # `|| cf_user=""` for the same pipefail reason as recorded_version: if the
      # az login has expired this pipeline would otherwise die right here,
      # silently, instead of falling through to the guidance just below.
      cf_user=$(az account show --query user.name -o tsv 2>/dev/null | cut -d@ -f1) || cf_user=""
      if [ -n "$cf_user" ] && [ -d "$cf_users/$cf_user" ]; then
        RESULTS_DIR=$cf_users/$cf_user/$share_subdir
      else
        log "Could not identify your folder under $cf_users (az user: ${cf_user:-unknown})"
        log "Set RESULTS_DIR explicitly to use the share, e.g.:"
        log "  RESULTS_DIR=$cf_users/<you>/$share_subdir $0"
      fi
    fi
  fi
  RESULTS_DIR=${RESULTS_DIR:-$REPO_ROOT/evaluate/results}
  mkdir -p "$RESULTS_DIR"
  log "Results will be written to $RESULTS_DIR"
  case "$RESULTS_DIR" in
    "$REPO_ROOT"/*) log "WARNING: results are on local disk - they will not survive instance deletion" ;;
  esac
}

# --- preflight: fail fast, before hours of training ---

# $@ are the sub-projects (train/merge/evaluate) this run needs.
pipeline_preflight() {
  local proj
  for proj in "$@"; do
    # Unconditional, not just when .venv is missing: a dependency added to a
    # pyproject.toml since the last run would otherwise go unnoticed until the
    # import error hours later. uv sync is a fast no-op when already current.
    log "Syncing $proj env"
    (cd "$REPO_ROOT/$proj" && uv sync)
  done

  # A CUDA GPU is not optional: the training configs are 4-bit (bitsandbytes),
  # and the merge step runs with --cuda. Checked here rather than discovered
  # after data prep, or worse, after an hour of something that looked like it
  # was working. ALLOW_NO_GPU=1 skips it, for exercising data prep on a laptop.
  if [ "${ALLOW_NO_GPU:-0}" != "1" ]; then
    # Probe via the evaluate env, not train: every pipeline needs evaluate, but
    # one that only merges and evaluates has no reason to sync train at all.
    if ! uv run --project "$REPO_ROOT/evaluate" python -c \
         'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
      echo "ERROR: no CUDA GPU visible - this pipeline cannot run here." >&2
      echo "Training is 4-bit (bitsandbytes) and merging runs with --cuda; neither" >&2
      echo "has a CPU or MPS path. This has to run ON a GPU compute instance, not" >&2
      echo "on a laptop that can reach one." >&2
      echo >&2
      # Named against the workspace actually in use - a hardcoded instance name
      # here sent someone to a stopped instance in a different workspace.
      echo "  az ml compute list --resource-group $RG --workspace-name $WS -o table" >&2
      echo "  az ml compute start --name <instance> --resource-group $RG --workspace-name $WS" >&2
      echo "  az ml compute connect-ssh --name <instance> --resource-group $RG --workspace-name $WS" >&2
      echo >&2
      echo "Then, on the instance (its prompt reads azureuser@<instance>):" >&2
      echo "  cd ~/model-merging && screen -S run $0" >&2
      echo >&2
      echo "Set ALLOW_NO_GPU=1 to skip this check (data prep only)." >&2
      exit 1
    fi
  fi

  uv run --project "$REPO_ROOT/evaluate" hf auth whoami >/dev/null 2>&1 \
    || die "Not logged into Hugging Face. Run: uv run --project evaluate hf auth login"

  if ! az ml model list --resource-group "$RG" --workspace-name "$WS" >/dev/null 2>&1; then
    echo "ERROR: 'az ml' is not working here - extension missing, or not logged in." >&2
    echo "On an Azure ML compute instance the system extension dir isn't writable, so:" >&2
    echo "  export AZURE_EXTENSION_DIR=\$HOME/.azure/cliextensions" >&2
    echo "  az extension add -n ml -y" >&2
    echo "  az login --use-device-code" >&2
    exit 1
  fi
}

# --- version bookkeeping ---
#
# Registration uses the next FREE version per name, never a hardcoded one:
# these names may already carry versions registered by other people, so
# hardcoding either collides with their work or silently skips registration
# and leaves evaluation measuring someone else's artifact. Versions this run
# creates are recorded in $VERSIONS_FILE, which is also what makes re-runs safe
# after cleanup: stages are skipped based on what has been REGISTERED, not on
# what happens to be on local disk, and anything needed again is re-fetched.

# Highest version currently registered for a model name, or 0 if none.
highest_version() {
  local name=$1 max
  # Same trap as recorded_version: `az ml model list --name X` exits 3 with
  # "Model container ... not found" when nothing has ever been registered under
  # that name, and pipefail carries that out of the pipeline. Treat a missing
  # container as version 0 rather than as an error.
  max=$(az ml model list --name "$name" --resource-group "$RG" --workspace-name "$WS" \
        --query "[].version" -o tsv 2>/dev/null | sort -n | tail -1) || max=""
  echo "${max:-0}"
}

# Version this pipeline run registered for a name, if any.
recorded_version() {
  [ -f "$VERSIONS_FILE" ] || return 0
  # The trailing `|| true` is load-bearing. grep exits 1 when the name isn't in
  # the file, which is the ordinary "not registered yet" case, and under
  # `set -o pipefail` that status becomes the pipeline's. Callers use this in a
  # plain assignment (existing=$(recorded_version ...)), so `set -e` would then
  # kill the script with no message at all - which it did, on every first-time
  # registration. Absence is an answer here, not a failure.
  grep "^$1=" "$VERSIONS_FILE" 2>/dev/null | tail -1 | cut -d= -f2 || true
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

# A directory is "the model" if it holds one of these.
model_dir_has_weights() {
  [ -f "$1/adapter_config.json" ] || [ -f "$1/config.json" ]
}

# `az ml model download` nests the artifact under --download-path, and how
# deeply has varied - a registered directory currently lands at
# <download-path>/<name>/<name>. Callers here use $path directly (the mergekit
# configs, Convert_to_full_model.py), so lift the real model directory's
# contents up to $path rather than making every call site guess the depth.
flatten_download() {
  local path=$1 marker inner
  model_dir_has_weights "$path" && return 0
  # Shallowest-first, and never a checkpoint directory. `find -print -quit`
  # returns whatever traversal order reaches first, and a training output dir
  # holds checkpoint-N/adapter_config.json alongside its own - so that flattened
  # an intermediate checkpoint over the real adapter, silently producing a model
  # directory that looked valid and was the wrong weights.
  marker=""
  for depth in 2 3 4; do
    marker=$( { find "$path" -mindepth "$depth" -maxdepth "$depth" \
                  \( -name adapter_config.json -o -name config.json \) 2>/dev/null \
                | grep -v "/checkpoint-" | head -1; } || true )
    [ -n "$marker" ] && break
  done
  [ -n "$marker" ] || return 1
  inner=$(dirname "$marker")
  log "Flattening nested download: $inner -> $path"
  ( shopt -s dotglob nullglob; mv "$inner"/* "$path"/ )
  find "$path" -mindepth 1 -type d -empty -delete
  model_dir_has_weights "$path"
}

# Make sure a registered artifact is present locally, downloading if cleanup
# (or a fresh instance) removed it.
ensure_local() {
  local name=$1 path=$2 v
  # Test for the model's own files rather than just the directory. A bare -d
  # test passes on an empty or nested directory left by an earlier attempt, and
  # the mistake then surfaces much later as a confusing error from peft or
  # mergekit about a missing config - which is exactly how the download nesting
  # bug in evaluate.py presented.
  if model_dir_has_weights "$path"; then
    return 0
  fi
  v=$(recorded_version "$name")
  [ -n "$v" ] || die "$path is missing and $name was not registered by this run"
  log "Fetching $name:$v from registry -> $path"
  az ml model download --name "$name" --version "$v" \
    --download-path "$(dirname "$path")" \
    --resource-group "$RG" --workspace-name "$WS" >/dev/null
  flatten_download "$path" \
    || die "Downloaded $name:$v but found no adapter_config.json or config.json anywhere under $path"
}
