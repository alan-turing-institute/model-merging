#!/usr/bin/env bash
# Evaluate handed-over models on XSum, through Inspect AI.
#
# For consuming someone else's trained artifacts off the workspace share rather
# than retraining them. Point it at a directory; it works out what each model is
# (adapter vs full weights, and an adapter's base from its own config rather
# than by guessing), evaluates every one on the same held-out test set, and
# writes the results JSONs the analysis tools already read.
#
#   ./run_handover_eval.sh
#   MODELS_DIR=~/cloudfiles/code/Users/jmcinroy/model-merging/models ./run_handover_eval.sh
#   EVAL_LIMIT=100 ./run_handover_eval.sh            # shakedown
#
# The base model is evaluated too, as the floor. Without it a table of handed-
# over models says which is best but not whether any of them beat doing nothing,
# and on XSum the untuned base is already a competent summariser - it scored
# ROUGE-1 0.281 - so that floor is high and easy to mistake for a result.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$PWD

MODELS_DIR=${MODELS_DIR:-$HOME/cloudfiles/code/Users/jmcinroy/model-merging/models}
DATASET=${DATASET:-../../datasets/xsum_dataset/test}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_LIMIT=${EVAL_LIMIT:-}
# Only set this to override what the adapters declare. Empty means "use each
# adapter's own base_model_name_or_path", which is the safe default.
BASE_MODEL=${BASE_MODEL:-}
RESULTS_SUBDIR=${RESULTS_SUBDIR:-model-merging-results/handover}

# A handover directory holds whatever its owner put there, including models for
# other tasks entirely. Evaluating a crime classifier on XSum produces a real
# number that means nothing, so select rather than take everything.
#   INCLUDE='llama3-xsum*' ./run_handover_eval.sh
INCLUDE=${INCLUDE:-*}
# Smoke runs live alongside real ones and are named accordingly. This default
# is a convenience, not a safeguard - check the step counts.
EXCLUDE=${EXCLUDE:-*-test}

source "$REPO_ROOT/pipeline_lib.sh"

pipeline_resolve_results_dir "$RESULTS_SUBDIR"

# No training, no merging, and nothing registered: this reads model directories
# off the share and writes results locally. The GPU, uv and HuggingFace checks
# all still apply - generation needs a GPU, and the base model is gated - but
# requiring an az login for it would block work that does not use the registry.
PIPELINE_NEEDS_AZ=0 pipeline_preflight evaluate

# The test split has to exist. On a fresh instance it does not, and preparing it
# is a minute; the alternative is a confusing load_from_disk failure several
# steps later.
if [ ! -d ../datasets/xsum_dataset ]; then
  log "Preparing XSum data (test split needed for evaluation)"
  uv run --project evaluate python prepare_xsum_data.py
fi

[ -d "$MODELS_DIR" ] || die "No such directory: $MODELS_DIR
  Set MODELS_DIR to where the handover actually landed. On a compute instance
  the share is under ~/cloudfiles/code/Users/<colleague>/..."

log "Discovering models under $MODELS_DIR"
DISCOVERED=$(cd evaluate && uv run python discover_models.py "$MODELS_DIR")
echo "$DISCOVERED" | while IFS=$'\t' read -r name kind path base; do
  log "  $name  ($kind, base=${base})"
done

evaluate_one() {
  local out=$1 model=$2 adapter=$3 provenance=$4
  local out_path=$RESULTS_DIR/$out
  if [ -f "$out_path" ]; then
    log "Already evaluated -> $out_path, skipping"
    return
  fi
  log "Evaluating -> $out_path"
  local args=(--model "$model" --dataset "$DATASET" --output "$out_path"
              --batch-size "$EVAL_BATCH_SIZE" --log-dir "$RESULTS_DIR/inspect-logs")
  if [ -n "$adapter" ]; then args+=(--adapter "$adapter"); fi
  if [ -n "$EVAL_LIMIT" ]; then args+=(--limit "$EVAL_LIMIT"); fi
  if [ -n "$provenance" ]; then args+=(--provenance "$provenance"); fi
  (cd evaluate && uv run python run_inspect_xsum.py "${args[@]}")
}

# Track the base models the adapters declare, so the floor is evaluated against
# the right one - and so a handover that mixes bases is visible rather than
# silently averaged into one table.
declared_bases=""

while IFS=$'\t' read -r name kind path base; do
  [ -n "$name" ] || continue
  # shellcheck disable=SC2254  # globs are intentional here
  case "$name" in
    $EXCLUDE) log "Skipping $name (matches EXCLUDE=$EXCLUDE)"; continue ;;
  esac
  case "$name" in
    $INCLUDE) ;;
    *) log "Skipping $name (does not match INCLUDE=$INCLUDE)"; continue ;;
  esac
  case "$kind" in
    adapter)
      model=${BASE_MODEL:-$base}
      [ -n "$model" ] && [ "$model" != "-" ] \
        || die "$name is an adapter but declares no base model; set BASE_MODEL"
      declared_bases="$declared_bases $model"
      evaluate_one "$name.json" "$model" "$path" "handover:$path"
      ;;
    full)
      # A full model needs no base to load, but config.json's _name_or_path
      # says what it was built from - which is the floor it should be read
      # against. Without that a distilled model's ROUGE has nothing to beat.
      if [ -n "$base" ] && [ "$base" != "-" ]; then
        declared_bases="$declared_bases $base"
      fi
      evaluate_one "$name.json" "$path" "" "handover:$path"
      ;;
    *) die "Unexpected kind '$kind' for $name" ;;
  esac
done <<< "$DISCOVERED"

# The floor, once per distinct base seen.
for model in $(echo "$declared_bases" | tr ' ' '\n' | sort -u); do
  [ -n "$model" ] || continue
  safe=$(echo "$model" | tr '/' '-')
  evaluate_one "base-$safe.json" "$model" "" "base:$model"
done

log "Done. Results in $RESULTS_DIR"
log "Compare:"
log "  uv run --project evaluate python experiments/bootstrap_rouge.py $RESULTS_DIR"
log "Per-sample view:"
log "  inspect view --log-dir $RESULTS_DIR/inspect-logs"
