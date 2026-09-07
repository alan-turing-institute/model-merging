#!/usr/bin/env bash
# Retrains each half-dataset adapter under several seeds, to establish how much
# a single half-data run varies before any merge comparison is trusted.
#
# Why this matters more than trying new merge methods: half 1 scored 92.1% and
# half 2 scored 95.8% on the same test set, from stratified random splits of
# identical size drawn from the same distribution. They should be roughly
# interchangeable. That 3.7pp gap is LARGER than the 1.2pp merge-vs-half-2 gap
# the interesting conclusion rests on, so run-to-run spread may swamp every
# effect measured so far. Until the spread is known, none of the differences
# are interpretable.
#
# The training configs set no `seed`, so every run so far used axolotl's
# default and the half-1/half-2 difference is attributable to which half of the
# data was used, not to training noise. This varies the seed with the data held
# fixed, which isolates training noise per half.
#
# Cheap by design: it evaluates each adapter as base_model+adapter, so nothing
# is converted to a full model and nothing is merged. Each adapter output_dir is
# ~540MB in practice (measured on upload) - the LoRA weights themselves are only
# ~25MB for 11.9M trainable params, the rest being tokenizer files, trainer
# state and checkpoints. So budget ~540MB per run locally plus the same again on
# the share, i.e. ~2GB local for the default two extra seeds across both halves.
# Pass MERGE_SEEDS=1 to additionally merge each seed's pair, which is where the
# disk cost lives (~26GB peak) - see experiments/merge_methods.sh for that.
#
# Usage, from the repo root:
#   experiments/seed_sweep.sh
#   SEEDS="43 44 45" experiments/seed_sweep.sh
#   MERGE_SEEDS=1 experiments/seed_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD

BASE_MODEL=google/gemma-3-4b-it
# 42 is axolotl's default and is what the existing runs used, so those results
# stand in as the seed-42 arm rather than being recomputed.
SEEDS=${SEEDS:-"43 44"}
RESULTS_DIR=${RESULTS_DIR:-$HOME/cloudfiles/code/Users/mpietrzyk/model-merging-results}

log() { echo "=== $* ===" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

mkdir -p "$RESULTS_DIR"

for half in 1 2; do
  [ -f "train/crime_gemma${half}.yaml" ] || die "Missing train/crime_gemma${half}.yaml"
done

for seed in $SEEDS; do
  for half in 1 2; do
    tag=${half}-of-2-seed${seed}
    config=train/crime_gemma${half}_seed${seed}.yaml
    outdir=models/gemma3-crime-${tag}-lora
    result=$RESULTS_DIR/half-${half}-seed${seed}.json

    if [ -f "$result" ]; then
      log "half $half seed $seed already evaluated, skipping"
      continue
    fi

    # Derive from the committed config so any change there carries over, only
    # overriding output_dir and appending the seed.
    if [ ! -f "$config" ]; then
      log "Writing $config"
      sed "s|^output_dir:.*|output_dir: ../${outdir}|" "train/crime_gemma${half}.yaml" > "$config"
      printf '\n# Set by experiments/seed_sweep.sh to measure run-to-run variance.\nseed: %s\n' "$seed" >> "$config"
    fi

    if [ -f "$outdir/adapter_model.safetensors" ] && [ -f "$outdir/trainer_state.json" ]; then
      log "half $half seed $seed already trained, skipping training"
    else
      log "Training half $half, seed $seed"
      (cd train && uv run axolotl train "$(basename "$config")")
      [ -f "$outdir/adapter_model.safetensors" ] || die "No adapter produced at $outdir"
    fi

    # Keep the adapter durable immediately - ~540MB each, and instances die.
    # Not silenced: this is the one artifact of the run worth preserving, so a
    # failure here should be visible rather than swallowed.
    share_dir=$(dirname "$RESULTS_DIR")
    if [ -d "$share_dir" ]; then
      cp -r "$outdir" "$share_dir/" || log "WARNING: failed to copy $outdir to $share_dir"
    else
      log "WARNING: $share_dir does not exist - adapter left only on local disk"
    fi

    log "Evaluating half $half, seed $seed"
    (cd evaluate && uv run python evaluate.py \
      --model "$BASE_MODEL" --adapter "../$outdir" --output "$result")
  done

  if [ "${MERGE_SEEDS:-0}" = "1" ]; then
    merged=models/gemma3-crime-merged-linear-seed${seed}
    result=$RESULTS_DIR/merged-linear-seed${seed}.json
    if [ -f "$result" ]; then
      log "seed $seed merge already evaluated, skipping"
    else
      config=merge/merge_linear_seed${seed}_config.yaml
      log "Writing $config"
      cat > "$config" << CFG
# Linear merge of the two half-dataset adapters trained with seed ${seed}.
# Written by experiments/seed_sweep.sh.
merge_method: linear
models:
  - model: "${BASE_MODEL}+../models/gemma3-crime-1-of-2-seed${seed}-lora"
    parameters:
      weight: 0.5
  - model: "${BASE_MODEL}+../models/gemma3-crime-2-of-2-seed${seed}-lora"
    parameters:
      weight: 0.5
tokenizer_source: union
dtype: float16
CFG
      log "Merging seed $seed"
      (cd merge && uv run mergekit-yaml "$(basename "$config")" "../$merged" \
        --cuda --lazy-unpickle --allow-crimes --lora-merge-cache "$REPO_ROOT/models/.lora_merge_cache")
      log "Evaluating seed $seed merge"
      (cd evaluate && uv run python evaluate.py --model "../$merged" --output "$result")
      rm -rf "$merged" "$REPO_ROOT/models/.lora_merge_cache"
    fi
  fi
done

log "Sweep complete. Results in $RESULTS_DIR:"
ls -1 "$RESULTS_DIR" >&2
cat >&2 << 'NOTE'

=== How to read this ===
Group the half-N-seed*.json results by half. The spread WITHIN a half is
training noise. Compare that spread against the 3.7pp between half 1 and
half 2, and against the 1.2pp between the merge and half 2:

  - if within-half spread is well under 1pp, both gaps are real effects
  - if it is comparable to 1-2pp, the merge-vs-half-2 comparison is noise and
    needs many more runs (or a paired test) before it means anything

Then run the significance tests:
  uv run --project evaluate python experiments/mcnemar.py RESULTS_DIR
NOTE
