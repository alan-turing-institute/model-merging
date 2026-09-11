#!/usr/bin/env bash
# Cross-task merging: does one model hold BOTH skills?
#
# The other two arms merge experts trained on the same task and ask whether
# merging recovers the full-data model. This asks the opposite question. The
# crime adapter classifies Reddit posts; the XSum adapter writes one-sentence
# summaries. Merge them and you either get a model that can do both, or the two
# task vectors interfere and you get one that does neither well.
#
# The merge is well-posed because the two arms happen to agree on everything
# that has to match: same base google/gemma-3-4b-it, same LoRA r=16/alpha=32,
# same q,k,v,o target modules. Only the training data differs, so each adapter
# is cleanly a task vector over one shared base.
#
# WHAT MAKES THIS READABLE: every model is evaluated on BOTH test sets, which
# gives four reference points per task rather than one.
#
#   base                floor - what the untuned model scores
#   its own expert      ceiling - what a model trained only on that task scores
#   the OTHER expert    the cross-task floor: what merging has to beat, and the
#                       number that says whether the other task's adapter is
#                       actively harmful rather than merely irrelevant
#   the merge           the experiment
#
# The failure mode is legible in a way it was not on the halves experiment:
# interference shows up as one task's output format leaking into the other.
# Watch num_unparsed in the crime results (the classifier rambling instead of
# emitting one word) and mean_pred_words in the XSum results (summaries
# collapsing toward "crime"/"not_crime"). Those two numbers say more about what
# a merge did than either headline metric.
#
# Usage, from the repo root on a GPU compute instance:
#   ./run_crosstask_pipeline.sh
#   MERGE_METHODS="task_arithmetic ties" ./run_crosstask_pipeline.sh
#   CRIME_LORA_VERSION=1 XSUM_LORA_VERSION=3 ./run_crosstask_pipeline.sh
#   EVAL_LIMIT=100 ./run_crosstask_pipeline.sh          # quick first pass
#   MERGE_CACHE_DIR=/tmp/lora-merge-cache-crosstask ./run_crosstask_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$PWD

# Azure ML registry to read from and write to. The compute instance does not
# have to live in this workspace - `az ml` talks to whichever one it is told to,
# so an instance in rg-tire-model-merging/tire-model-merging can still use the
# artifacts registered in tire-1/tire-2. Override to target a different one:
#   AZUREML_RG=rg-tire-model-merging AZUREML_WS=tire-model-merging ./run_crosstask_pipeline.sh
RG=${AZUREML_RG:-tire-1}
WS=${AZUREML_WS:-tire-2}
BASE_MODEL=google/gemma-3-4b-it
VERSIONS_FILE=$REPO_ROOT/.pipeline_versions_crosstask

# Which registered adapters to merge. Both names carry several versions from
# repeated runs of their own arms, and picking the wrong one silently measures
# a different experiment - so they are explicit rather than "whatever is latest".
CRIME_LORA_VERSION=${CRIME_LORA_VERSION:-3}
XSUM_LORA_VERSION=${XSUM_LORA_VERSION:-3}

MERGE_METHODS=${MERGE_METHODS:-"linear task_arithmetic ties"}
MERGE_CACHE_DIR=${MERGE_CACHE_DIR:-$REPO_ROOT/models/.lora_merge_cache_crosstask}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_LIMIT=${EVAL_LIMIT:-}
# inspect (default) or legacy - see run_xsum_pipeline.sh. Both write the same
# results JSON, which is what crosstask_table.py reads.
XSUM_EVAL=${XSUM_EVAL:-inspect}
# Same choice for the classification side - see run_kd_pipeline.sh.
CRIME_EVAL=${CRIME_EVAL:-inspect}
# Each merge is ~8.1GB and there are several. They are registered, then the
# local copy is deleted once both evaluations are done, so peak disk is the
# cache plus ONE merge rather than the cache plus all of them.
KEEP_MERGES=${KEEP_MERGES:-0}
REGISTER_MERGES=${REGISTER_MERGES:-1}

source "$REPO_ROOT/pipeline_lib.sh"

pipeline_resolve_results_dir model-merging-results/crosstask
pipeline_preflight merge evaluate

mkdir -p models
mkdir -p "$MERGE_CACHE_DIR" 2>/dev/null || die "Cannot create MERGE_CACHE_DIR=$MERGE_CACHE_DIR
  /mnt is root-owned on an Azure ML compute instance; /tmp is the same
  filesystem and is writable:
    MERGE_CACHE_DIR=/tmp/lora-merge-cache-crosstask $0"

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
avail=$(free_gb "$MERGE_CACHE_DIR")
log "Free space at $MERGE_CACHE_DIR: ${avail:-?}GB (needs ~17GB cache + ~9GB per merge)"

# --- 1. both datasets ---

if [ ! -d ../datasets/crime_dataset ]; then
  log "Preparing crime data"
  uv run --project evaluate python prepare_data.py
fi
if [ ! -d ../datasets/xsum_dataset ]; then
  log "Preparing XSum data"
  uv run --project evaluate python prepare_xsum_data.py
fi

# --- 2. fetch the two experts at the requested versions ---
# Not ensure_local: that resolves versions this run registered, and these were
# registered by the other two arms.

fetch_model() {
  local name=$1 version=$2 path=$3
  if model_dir_has_weights "$path"; then
    log "$name already present at $path, skipping download"
    return
  fi
  log "Fetching $name:$version -> $path"
  az ml model download --name "$name" --version "$version" \
    --download-path "$(dirname "$path")" \
    --resource-group "$RG" --workspace-name "$WS" >/dev/null
  flatten_download "$path" \
    || die "Downloaded $name:$version but found no adapter_config.json under $path"
}

CRIME_LORA=models/gemma3-crime-full-lora
XSUM_LORA=models/gemma3-xsum-full-lora
fetch_model gemma3-crime-full-lora "$CRIME_LORA_VERSION" "$CRIME_LORA"
fetch_model gemma3-xsum-full-lora  "$XSUM_LORA_VERSION"  "$XSUM_LORA"

log "Merging azureml:gemma3-crime-full-lora:$CRIME_LORA_VERSION"
log "   with azureml:gemma3-xsum-full-lora:$XSUM_LORA_VERSION"

# --- 3. evaluation on both tasks ---
# Results are named <stem>.<task>.json so experiments/crosstask_table.py can
# pair a model's two scores back together.

eval_crime() {
  local stem=$1 model=$2 adapter=$3 provenance=$4
  local out=$RESULTS_DIR/$stem.crime.json
  if [ -f "$out" ]; then log "Already evaluated -> $out, skipping"; return; fi
  log "Evaluating CRIME -> $out (model=$model adapter=${adapter:-none})"
  local args=(--model "$model" --output "$out")
  if [ -n "$adapter" ]; then args+=(--adapter "$adapter"); fi
  if [ -n "$EVAL_LIMIT" ]; then args+=(--limit "$EVAL_LIMIT"); fi
  if [ -n "$provenance" ]; then args+=(--provenance "$provenance"); fi
  if [ "$CRIME_EVAL" = "inspect" ]; then
    args+=(--batch-size "$EVAL_BATCH_SIZE" --log-dir "$RESULTS_DIR/inspect-logs")
    (cd evaluate && uv run python run_inspect_crime.py "${args[@]}")
  else
    (cd evaluate && uv run python evaluate.py "${args[@]}")
  fi
}

eval_xsum() {
  local stem=$1 model=$2 adapter=$3 provenance=$4
  local out=$RESULTS_DIR/$stem.xsum.json
  if [ -f "$out" ]; then log "Already evaluated -> $out, skipping"; return; fi
  log "Evaluating XSUM -> $out (model=$model adapter=${adapter:-none})"
  local args=(--model "$model" --output "$out" --batch-size "$EVAL_BATCH_SIZE")
  if [ -n "$adapter" ]; then args+=(--adapter "$adapter"); fi
  if [ -n "$EVAL_LIMIT" ]; then args+=(--limit "$EVAL_LIMIT"); fi
  if [ -n "$provenance" ]; then args+=(--provenance "$provenance"); fi
  if [ "$XSUM_EVAL" = "inspect" ]; then
    args+=(--log-dir "$RESULTS_DIR/inspect-logs")
    (cd evaluate && uv run python run_inspect_xsum.py "${args[@]}")
  else
    (cd evaluate && uv run python evaluate_summarisation.py "${args[@]}")
  fi
}

eval_both() {
  eval_crime "$@"
  eval_xsum "$@"
}

# --- 4. the four reference points ---

eval_both base "$BASE_MODEL" "" ""
eval_both crime-expert "$BASE_MODEL" "$REPO_ROOT/$CRIME_LORA" \
  "azureml:gemma3-crime-full-lora:$CRIME_LORA_VERSION"
eval_both xsum-expert  "$BASE_MODEL" "$REPO_ROOT/$XSUM_LORA" \
  "azureml:gemma3-xsum-full-lora:$XSUM_LORA_VERSION"

# --- 5. merge, evaluate, and free the merge before the next one ---

for method in $MERGE_METHODS; do
  config=merge/merge_${method}_crosstask_config.yaml
  [ -f "$config" ] || die "No merge config at $config (MERGE_METHODS=$MERGE_METHODS)"
  name=gemma3-crosstask-merged-${method//_/-}
  outdir=models/$name

  # The real resume condition is "both evaluations exist" - the merge itself is
  # a regenerable intermediate that gets deleted after use.
  if [ -f "$RESULTS_DIR/$name.crime.json" ] && [ -f "$RESULTS_DIR/$name.xsum.json" ]; then
    log "$name already evaluated on both tasks, skipping"
    continue
  fi

  if [ -f "$outdir/.merge_complete" ]; then
    log "$name already merged locally, skipping merge"
  else
    rm -rf "$outdir"   # clear any partial output from an interrupted attempt
    log "Merging ($method): crime + xsum"
    (cd merge && uv run mergekit-yaml "$(basename "$config")" "../$outdir" \
       --cuda --lazy-unpickle --allow-crimes \
       --lora-merge-cache "$MERGE_CACHE_DIR")
    touch "$outdir/.merge_complete"
  fi

  provenance="local:$name(crime:$CRIME_LORA_VERSION+xsum:$XSUM_LORA_VERSION)"
  if [ "$REGISTER_MERGES" = "1" ]; then
    register_model "$name" "$outdir"
    provenance=$(ref "$name")
  fi

  eval_both "$name" "$REPO_ROOT/$outdir" "" "$provenance"

  if [ "$KEEP_MERGES" != "1" ]; then
    log "Freeing $outdir ($(du -sh "$outdir" 2>/dev/null | cut -f1)) - evaluated, and registered if REGISTER_MERGES=1"
    rm -rf "$outdir"
  fi
done

log "Done. Results in $RESULTS_DIR"
log "Joint table:"
log "  uv run --project evaluate python experiments/crosstask_table.py $RESULTS_DIR"
if [ "$MERGE_CACHE_DIR" != "$REPO_ROOT/models/.lora_merge_cache_crosstask" ]; then
  log "Merge cache left at $MERGE_CACHE_DIR - remove it by hand"
fi
