#!/usr/bin/env bash
# XSum summarisation arm of the disjoint-halves merging experiment.
#
# Same question as the crime-classification arm, on a generation task: if you
# split a training set in two, fine-tune an expert on each half in parallel,
# and merge the two experts, do you get back the model you would have got by
# training on all the data at once? ("Poor man's parallelism.")
#
# What gets trained and compared:
#
#   gemma3-xsum-full-lora     LoRA on ALL the training data      - the target
#   gemma3-xsum-1-of-2-lora   LoRA on a disjoint random half     - merge input
#   gemma3-xsum-2-of-2-lora   LoRA on the other half             - merge input
#   gemma3-xsum-merged-<m>    the two halves merged with method m - the result
#
# plus the untouched base model, evaluated zero-shot. That last one matters
# more here than on the classification arm: a merge that has drifted back
# towards base behaviour is the specific failure this experiment is looking
# for, so the base model's own ROUGE is the floor to read the merge against.
#
# The merge configs use mergekit's "<base>+<adapter>" syntax, so the two
# half-data adapters are not converted into full models as a separate step.
# That saves the conversion step, but NOT the disk: --lora-merge-cache is where
# mergekit materialises each combination, as a full model each (~8.1GB for
# gemma-3-4b), so a two-way merge still needs ~17GB of cache plus ~8.1GB per
# output. The cache is identical across merge methods, so it is built once and
# reused by every entry in MERGE_METHODS.
#
# Disk is the binding constraint on an Azure ML compute instance, where the
# root disk is often tighter than the temp disk. Two things follow: the
# full-data model's local copy is deleted as soon as it is registered (step 4),
# and MERGE_CACHE_DIR can put the cache on another filesystem entirely.
#
# Storage policy: local disk is SCRATCH ONLY. Adapters, the full-data model and
# the merges go to the Azure ML registry; results JSONs go to the workspace
# share when it is mounted; everything else is deleted at the end. Set
# KEEP_LOCAL=1 to keep models/ for debugging.
#
# Usage: run from the repo root on a GPU compute instance, with `az` working
# and `hf auth login` already done. Use screen/tmux - this takes hours.
#
#   ./run_xsum_pipeline.sh
#   MERGE_METHODS="linear task_arithmetic ties" ./run_xsum_pipeline.sh
#   XSUM_TRAIN_N=1000 XSUM_TEST_N=100 EVAL_LIMIT=50 ./run_xsum_pipeline.sh  # smoke test
#   RESULTS_DIR=/some/path ./run_xsum_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$PWD

RG=tire-1
WS=tire-2
BASE_MODEL=google/gemma-3-4b-it
# Separate from the crime arm's .pipeline_versions so the two don't collide.
VERSIONS_FILE=$REPO_ROOT/.pipeline_versions_xsum

# Which merges to run. Each name maps to merge/merge_<name>_xsum_config.yaml
# and registers as gemma3-xsum-merged-<name with underscores as dashes>.
MERGE_METHODS=${MERGE_METHODS:-linear}

# Where mergekit materialises each base+adapter combination. Needs ~17GB for a
# two-way merge - the single largest disk consumer in the pipeline. Put it on
# another filesystem when the repo's disk is tight:
#   MERGE_CACHE_DIR=/mnt/lora-merge-cache ./run_xsum_pipeline.sh
MERGE_CACHE_DIR=${MERGE_CACHE_DIR:-$REPO_ROOT/models/.lora_merge_cache}

# Passed through to evaluate_summarisation.py. EVAL_LIMIT caps the number of
# test examples, for smoke tests.
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_LIMIT=${EVAL_LIMIT:-}

source "$REPO_ROOT/pipeline_lib.sh"

pipeline_resolve_results_dir model-merging-results/xsum
pipeline_preflight train merge evaluate

# --- 1. data prep ---

if [ ! -d ../datasets/xsum_dataset ]; then
  log "Preparing XSum data"
  uv run --project evaluate python prepare_xsum_data.py
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
  # axolotl's output. Inferring is what broke here before: the check looked for
  # trainer_state.json at the top of output_dir, but axolotl only writes that
  # inside checkpoint-N/, so the condition was never true and every
  # unregistered adapter retrained from scratch. adapter_model.safetensors on
  # its own is no good either - it is pre-saved before step 0, so it exists for
  # an interrupted run too.
  if [ -f "$outdir/.training_complete" ]; then
    log "Adapter already trained at $outdir, skipping"
    return
  fi
  log "Training $cfg"
  (cd train && uv run axolotl train "$cfg")
  [ -f "$outdir/adapter_model.safetensors" ] || die "Training finished but no adapter at $outdir"
  touch "$outdir/.training_complete"
}
train_if_needed xsum_gemma.yaml  gemma3-xsum-full-lora   models/gemma3-xsum-full-lora
train_if_needed xsum_gemma1.yaml gemma3-xsum-1-of-2-lora models/gemma3-xsum-1-of-2-lora
train_if_needed xsum_gemma2.yaml gemma3-xsum-2-of-2-lora models/gemma3-xsum-2-of-2-lora

# --- 3. register the adapters (the trained output - this must land) ---

register_model gemma3-xsum-full-lora   models/gemma3-xsum-full-lora
register_model gemma3-xsum-1-of-2-lora models/gemma3-xsum-1-of-2-lora
register_model gemma3-xsum-2-of-2-lora models/gemma3-xsum-2-of-2-lora

# --- 4. materialise the full-data model ---
# The baseline the merge is compared against, and the one place where
# base+adapter and merged weights can be checked to agree.

if [ -n "$(recorded_version gemma3-xsum-full)" ]; then
  log "Full model already registered, skipping conversion"
elif [ -f models/gemma3-xsum-full/config.json ]; then
  log "Full model already converted, skipping"
else
  ensure_local gemma3-xsum-full-lora models/gemma3-xsum-full-lora
  log "Converting gemma3-xsum-full-lora -> models/gemma3-xsum-full"
  uv run --project evaluate python Convert_to_full_model.py \
    "$BASE_MODEL" models/gemma3-xsum-full-lora models/gemma3-xsum-full
fi
register_model gemma3-xsum-full models/gemma3-xsum-full

# Drop the local copy immediately: it is durable in the registry, evaluation
# references it as azureml:<name>:<version> rather than by path, and the merge
# below needs every GB it can get. Holding a third full model on local disk
# through the merge is what filled the root disk on the first real run.
if [ "${KEEP_LOCAL:-0}" != "1" ] && [ -d models/gemma3-xsum-full ]; then
  log "Freeing models/gemma3-xsum-full ($(du -sh models/gemma3-xsum-full | cut -f1)) - registered, and evaluation pulls it from the registry"
  rm -rf models/gemma3-xsum-full
fi

# --- 5. merge the two half-data adapters ---

mkdir -p models
# A caller-supplied cache path may not be creatable - /mnt on an Azure ML
# compute instance is root-owned, so MERGE_CACHE_DIR=/mnt/... fails here with a
# bare mkdir error and no hint about what to do instead.
mkdir -p "$MERGE_CACHE_DIR" 2>/dev/null || die "Cannot create MERGE_CACHE_DIR=$MERGE_CACHE_DIR
  On an Azure ML compute instance /mnt is root-owned, but /tmp sits on the SAME
  filesystem and is writable, so prefer:
    MERGE_CACHE_DIR=/tmp/lora-merge-cache $0
  Or create the directory once, with sudo:
    sudo mkdir -p $MERGE_CACHE_DIR && sudo chown \$USER:\$USER $MERGE_CACHE_DIR"

# Fail early and legibly rather than part-way through a merge - which is
# exactly how the first real run ended, with safetensors hitting StorageFull
# after two hours of training and uploads.
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
fs_of()   { df -P "$1" 2>/dev/null | tail -1 | awk '{print $1}'; }

# Cache is only a cost if it still has to be built; it is reused across methods.
CACHE_GB=18
if [ -n "$(ls -A "$MERGE_CACHE_DIR" 2>/dev/null)" ]; then
  CACHE_GB=0
fi
OUTPUT_GB=10

cache_free=$(free_gb "$MERGE_CACHE_DIR")
out_free=$(free_gb "$REPO_ROOT")
log "Free space: ${cache_free:-?}GB at $MERGE_CACHE_DIR (cache), ${out_free:-?}GB at $REPO_ROOT (output)"

space_advice="  Put the cache on another filesystem:
    MERGE_CACHE_DIR=/mnt/lora-merge-cache $0
  or reclaim space:
    rm -rf ~/.cache/huggingface/hub   # forces an 8.1GB base-model re-download later"

if [ "$(fs_of "$MERGE_CACHE_DIR")" = "$(fs_of "$REPO_ROOT")" ]; then
  # One filesystem: the cache and the merge output compete for the same bytes,
  # so the requirement is their SUM. Checking each against its own threshold
  # would have passed the 24GB-free state that this run actually died on.
  need=$(( CACHE_GB + OUTPUT_GB ))
  if [ -n "$out_free" ] && [ "$out_free" -lt "$need" ]; then
    die "Cache and output share one filesystem, so this needs ~${need}GB free at
  $REPO_ROOT, and there is ${out_free}GB.
$space_advice"
  fi
else
  if [ -n "$cache_free" ] && [ "$cache_free" -lt "$CACHE_GB" ]; then
    die "Need ~${CACHE_GB}GB free at $MERGE_CACHE_DIR for the adapter cache, have ${cache_free}GB.
$space_advice"
  fi
  if [ -n "$out_free" ] && [ "$out_free" -lt "$OUTPUT_GB" ]; then
    die "Need ~${OUTPUT_GB}GB free at $REPO_ROOT for each merge output, have ${out_free}GB."
  fi
fi

merged_names=()
for method in $MERGE_METHODS; do
  config=merge/merge_${method}_xsum_config.yaml
  [ -f "$config" ] || die "No merge config at $config (MERGE_METHODS=$MERGE_METHODS)"
  name=gemma3-xsum-merged-${method//_/-}
  outdir=models/$name
  merged_names+=("$name")

  if [ -n "$(recorded_version "$name")" ]; then
    log "$name already registered, skipping merge"
  elif [ -f "$outdir/config.json" ]; then
    log "$name already merged locally, skipping"
  else
    # The merge configs read the adapters from ../models/, so fetch them back
    # if a previous run's cleanup removed them. Only when a merge is actually
    # about to run - re-running a fully-registered pipeline shouldn't download
    # anything.
    ensure_local gemma3-xsum-1-of-2-lora models/gemma3-xsum-1-of-2-lora
    ensure_local gemma3-xsum-2-of-2-lora models/gemma3-xsum-2-of-2-lora
    log "Merging ($method)"
    # --lora-merge-cache is required by the "<base>+<adapter>" model syntax:
    # it's where mergekit materialises each combination before merging. The
    # cache is shared across methods, so later merges reuse the first one's
    # work. --allow-crimes permits merges mergekit considers unusual;
    # --lazy-unpickle keeps peak memory down by memory-mapping the weights.
    (cd merge && uv run mergekit-yaml "$(basename "$config")" "../$outdir" \
       --cuda --lazy-unpickle --allow-crimes \
       --lora-merge-cache "$MERGE_CACHE_DIR")
  fi
  register_model "$name" "$outdir"
done

# --- 6. evaluate ---
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
  local args=(--model "$model" --output "$out_path" --batch-size "$EVAL_BATCH_SIZE")
  if [ -n "$adapter" ]; then args+=(--adapter "$adapter"); fi
  if [ -n "$EVAL_LIMIT" ]; then args+=(--limit "$EVAL_LIMIT"); fi
  (cd evaluate && uv run python evaluate_summarisation.py "${args[@]}")
}

# The floor: what the base model scores with no fine-tuning at all.
evaluate_if_needed "$BASE_MODEL" "" gemma3-xsum-base.json
# The target: all the data, one training run.
evaluate_if_needed "$BASE_MODEL" "$(ref gemma3-xsum-full-lora)" gemma3-xsum-full-lora.json
# The same thing materialised into full weights - should match the line above,
# and a divergence means merge_and_unload() changed behaviour.
evaluate_if_needed "$(ref gemma3-xsum-full)" "" gemma3-xsum-full.json
# The merge's own inputs. Without these there is nothing to say whether a merge
# beat its parts or just landed between them - which was the whole finding on
# the classification arm.
evaluate_if_needed "$BASE_MODEL" "$(ref gemma3-xsum-1-of-2-lora)" gemma3-xsum-1-of-2-lora.json
evaluate_if_needed "$BASE_MODEL" "$(ref gemma3-xsum-2-of-2-lora)" gemma3-xsum-2-of-2-lora.json
# The experiment.
for name in "${merged_names[@]}"; do
  evaluate_if_needed "$(ref "$name")" "" "$name.json"
done

# --- 7. clean up local scratch ---

if [ "${KEEP_LOCAL:-0}" = "1" ]; then
  log "KEEP_LOCAL=1 - leaving models/ in place ($(du -sh models 2>/dev/null | cut -f1) on disk)"
else
  log "Removing local scratch models/ ($(du -sh models 2>/dev/null | cut -f1)) - everything durable is registered"
  rm -rf models
  # A cache outside the repo is not covered by removing models/. It is NOT
  # deleted automatically: MERGE_CACHE_DIR is caller-supplied, and rm -rf on a
  # path this script did not choose is not a risk worth taking to save a step.
  case "$MERGE_CACHE_DIR" in
    "$REPO_ROOT"/models/*) ;;
    *) log "Merge cache left at $MERGE_CACHE_DIR ($(du -sh "$MERGE_CACHE_DIR" 2>/dev/null | cut -f1)) - remove it by hand" ;;
  esac
fi

log "Pipeline complete. Versions registered by this run:"
cat "$VERSIONS_FILE" >&2
log "Results in $RESULTS_DIR"
log "Compare them with:"
log "  uv run --project evaluate python experiments/bootstrap_rouge.py $RESULTS_DIR"
log "Note: the base model cache in ~/.cache/huggingface (~8.6GB) is left in place;"
log "clear it with 'rm -rf ~/.cache/huggingface/hub' if you need the space back."
