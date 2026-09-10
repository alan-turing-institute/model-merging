#!/usr/bin/env bash
# Copy registered models from one Azure ML workspace to another.
#
# Written to move this project's adapters from tire-1/tire-2 into the dedicated
# rg-tire-model-merging/tire-model-merging workspace without retraining them.
#
# ONLY THE LoRA ADAPTERS ARE WORTH COPYING (~370MB each). The full-data model
# and the merges are ~8.1GB each and are regenerable: run_xsum_pipeline.sh
# reconverts the full model from its adapter and re-runs the merges, and both
# get registered in the destination as a side effect. Copying them would move
# ~25GB to save minutes of compute.
#
# WHICH VERSIONS: registry versions accumulate across runs, and several of
# these names carry smoke-test artifacts. The defaults below are the real
# full-scale ones, identified by registration time:
#
#   gemma3-xsum-full-lora    v1 19:17, v2 19:51 = smoke; v3 20:34 = real
#   gemma3-xsum-1-of-2-lora  v1 19:58 = smoke;   v2 20:35 = real
#   gemma3-xsum-2-of-2-lora  v1 19:58 = smoke;   v2 20:35 = real
#   gemma3-crime-full-lora   v3 is the most recent; v1/v2 predate this work
#                            by a week and their provenance is NOT established
#
# After copying, the destination versions are recorded in VERSIONS_FILE, which
# is what makes the pipeline skip straight past training: it treats those names
# as already registered by the run, and `ref` resolves them in the destination.
#
# Usage, from the repo root:
#   ./migrate_models.sh
#   MODELS="gemma3-xsum-full-lora:3" ./migrate_models.sh
#   DST_WS=some-other-workspace ./migrate_models.sh
#
# Then run the pipeline against the destination:
#   AZUREML_RG=rg-tire-model-merging AZUREML_WS=tire-model-merging \
#     ./run_xsum_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$PWD

SRC_RG=${SRC_RG:-tire-1}
SRC_WS=${SRC_WS:-tire-2}
DST_RG=${DST_RG:-rg-tire-model-merging}
DST_WS=${DST_WS:-tire-model-merging}

MODELS=${MODELS:-"gemma3-xsum-full-lora:3 gemma3-xsum-1-of-2-lora:2 gemma3-xsum-2-of-2-lora:2 gemma3-crime-full-lora:3"}
VERSIONS_FILE=${VERSIONS_FILE:-$REPO_ROOT/.pipeline_versions_xsum}

# log/die/model_dir_has_weights/flatten_download. The registry helpers in there
# read $RG/$WS, which is why this script does not use them - it talks to two
# workspaces and passes them explicitly.
source "$REPO_ROOT/pipeline_lib.sh"

log "Source     : $SRC_RG / $SRC_WS"
log "Destination: $DST_RG / $DST_WS"
log "Versions   : $VERSIONS_FILE"

az ml model list --resource-group "$DST_RG" --workspace-name "$DST_WS" >/dev/null 2>&1 \
  || die "Cannot reach $DST_RG/$DST_WS with 'az ml'. Check the names and that you have access."

mkdir -p models

dst_highest_version() {
  local name=$1 max
  max=$(az ml model list --name "$name" --resource-group "$DST_RG" --workspace-name "$DST_WS" \
        --query "[].version" -o tsv 2>/dev/null | sort -n | tail -1) || max=""
  echo "${max:-0}"
}

already_recorded() {
  [ -f "$VERSIONS_FILE" ] || return 1
  grep -q "^$1=" "$VERSIONS_FILE" 2>/dev/null
}

for entry in $MODELS; do
  name=${entry%%:*}
  version=${entry##*:}
  [ "$name" != "$version" ] || die "MODELS entries must be name:version, got '$entry'"
  path=models/$name

  if already_recorded "$name"; then
    log "$name already recorded in $VERSIONS_FILE, skipping"
    continue
  fi

  # The destination may already have this model - from an earlier run of this
  # script, possibly on another machine, since it is all just az calls. Reuse
  # that registration rather than uploading a second identical copy: the
  # versions file and the local models/ directory are per-machine, so running
  # this on a second box must not multiply registry versions.
  dst_existing=$(dst_highest_version "$name")

  if [ "$dst_existing" != "0" ] && [ "${FORCE_REREGISTER:-0}" != "1" ]; then
    next=$dst_existing
    log "$name already in $DST_WS as version $next - reusing (FORCE_REREGISTER=1 to re-upload)"
    if model_dir_has_weights "$path"; then
      log "$name already present at $path, skipping download"
    else
      # Pull from the destination, not the source: same bytes, and it keeps
      # this path working even without access to the source workspace.
      log "Downloading $name:$next from $DST_WS"
      az ml model download --name "$name" --version "$next" \
        --download-path models \
        --resource-group "$DST_RG" --workspace-name "$DST_WS" >/dev/null
      flatten_download "$path" \
        || die "Downloaded $name:$next but found no adapter_config.json under $path"
    fi
  else
    if model_dir_has_weights "$path"; then
      log "$name already present at $path, skipping download"
    else
      log "Downloading $name:$version from $SRC_WS"
      az ml model download --name "$name" --version "$version" \
        --download-path models \
        --resource-group "$SRC_RG" --workspace-name "$SRC_WS" >/dev/null
      flatten_download "$path" \
        || die "Downloaded $name:$version but found no adapter_config.json under $path"
    fi

    next=$(( dst_existing + 1 ))
    log "Registering $name as version $next in $DST_WS"
    az ml model create --name "$name" --version "$next" --type custom_model \
      --path "$path" --resource-group "$DST_RG" --workspace-name "$DST_WS" >/dev/null
  fi

  # 3. record it, so the pipeline treats this name as done and `ref` resolves
  echo "$name=$next" >> "$VERSIONS_FILE"

  # 4. adapters carry the training sentinel, so train_if_needed skips them even
  #    if the versions file is later cleared
  if [ -f "$path/adapter_config.json" ]; then
    touch "$path/.training_complete"
  fi

  log "$name: $SRC_WS:$version -> $DST_WS:$next"
done

log "Done. Recorded in $VERSIONS_FILE:"
cat "$VERSIONS_FILE" >&2
log "Now run, against the destination workspace:"
log "  AZUREML_RG=$DST_RG AZUREML_WS=$DST_WS ./run_xsum_pipeline.sh"
