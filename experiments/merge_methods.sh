#!/usr/bin/env bash
# Sweeps every merge method in merge/ over the same two half-dataset adapters,
# evaluating each result on the test set.
#
# Why: a plain `linear` merge at 0.5/0.5 produced base + 0.5*d1 + 0.5*d2, which
# scored BELOW the better of its two inputs (94.6% vs 95.8%), losing almost all
# of it in recall while precision held. That pattern is what under-adaptation
# looks like - halving both deltas pulls the model back toward base behaviour,
# so it stops confidently predicting the positive class. These methods test
# alternatives: combining deltas at full strength, sparsifying them, taking
# sign consensus, or interpolating on the sphere instead of the straight line.
#
# Disk: every config here uses mergekit's base_model+adapter syntax, so each
# combination has to be materialised into --lora-merge-cache first (~8.6GB
# each, ~17GB total). That cache is IDENTICAL for every method, so it is built
# once and reused; only each method's ~8.6GB output is transient and is deleted
# after evaluation. Peak usage is therefore ~26GB, not ~26GB per method.
#
# Usage, from the repo root:
#   experiments/merge_methods.sh
#   KEEP_MERGES=1 experiments/merge_methods.sh   # don't delete each merge
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD

BASE_MODEL=google/gemma-3-4b-it
CACHE=$REPO_ROOT/models/.lora_merge_cache
RESULTS_DIR=${RESULTS_DIR:-$HOME/cloudfiles/code/Users/mpietrzyk/model-merging-results}

log() { echo "=== $* ===" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

mkdir -p "$RESULTS_DIR" models

# Fail early and legibly rather than dying part-way through a merge, which is
# how the first attempt at this went.
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
avail=$(free_gb "$REPO_ROOT")
[ -n "$avail" ] || die "Could not determine free space for $REPO_ROOT"
log "Free space: ${avail}GB"
if [ "$avail" -lt 30 ]; then
  die "Need ~30GB free (17GB adapter cache + 8.6GB per merge). Free space first:
  rm -rf models/gemma3-crime-full models/gemma3-crime-1-of-2 models/gemma3-crime-2-of-2
  rm -rf ~/.cache/huggingface/hub   # forces an 8.6GB re-download later"
fi

for a in models/gemma3-crime-1-of-2-lora models/gemma3-crime-2-of-2-lora; do
  [ -d "$a" ] || die "Missing adapter $a - restore it from the share or the registry first"
done

# A weight-1.0 task-arithmetic variant, which is not in merge/ but is the
# direct test of the under-adaptation hypothesis above: keep both deltas at
# full strength instead of halving them. Expect this to overshoot if the
# deltas reinforce, so it is informative either way.
cat > merge/merge_task_arithmetic_w1_config.yaml << 'CFG'
# Task arithmetic with weight 1.0 on both deltas: base + d1 + d2 rather than
# base + 0.5*d1 + 0.5*d2. Tests whether the plain linear merge underperformed
# because averaging halved the task adaptation.
merge_method: task_arithmetic
base_model: "google/gemma-3-4b-it"
models:
  - model: "google/gemma-3-4b-it+../models/gemma3-crime-1-of-2-lora"
    parameters:
      weight: 1.0
  - model: "google/gemma-3-4b-it+../models/gemma3-crime-2-of-2-lora"
    parameters:
      weight: 1.0
tokenizer_source: union
dtype: float16
CFG

METHODS=(
  task_arithmetic
  task_arithmetic_w1
  ties
  dare_ties
  model_stock
  slerp
  arcee_fusion
)

for method in "${METHODS[@]}"; do
  config=merge/merge_${method}_config.yaml
  out=models/gemma3-crime-merged-$method
  result=$RESULTS_DIR/merged-$method.json

  if [ ! -f "$config" ]; then
    log "No config at $config, skipping $method"
    continue
  fi
  if [ -f "$result" ]; then
    log "$method already evaluated, skipping"
    continue
  fi

  log "Merging: $method"
  # Some methods legitimately fail on some inputs (model_stock needs 3+ models
  # to estimate its angle, slerp takes exactly 2). Don't abort the whole sweep.
  if ! (cd merge && uv run mergekit-yaml "merge_${method}_config.yaml" "../$out" \
        --cuda --lazy-unpickle --allow-crimes --lora-merge-cache "$CACHE"); then
    log "$method FAILED to merge - continuing with the rest"
    rm -rf "$out"
    continue
  fi

  log "Evaluating: $method"
  (cd evaluate && uv run python evaluate.py --model "../$out" --output "$result")

  if [ "${KEEP_MERGES:-0}" = "1" ]; then
    log "KEEP_MERGES=1 - retaining $out"
  else
    log "Removing $out (result saved to $result)"
    rm -rf "$out"
  fi
done

log "Sweep complete. Removing the shared adapter cache."
rm -rf "$CACHE"

log "Results in $RESULTS_DIR:"
ls -1 "$RESULTS_DIR" >&2
log "Compare them with: uv run --project evaluate python experiments/mcnemar.py $RESULTS_DIR"
