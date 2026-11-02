#!/bin/bash
# Submits one repeat of the crime experiments, trained and merged with seed S,
# as dependent chains:
#   full model                    -> eval
#   each part of each N-way split -> eval each part
#                                 -> 7 merges of the N parts -> eval each merge
# for each N given (default 2 4 8). The data is the same for every seed (the
# prep seed and the contiguous splits are fixed), so repeats differ only in
# training (data order, dropout, LoRA initialisation) and in dare_ties' random
# dropping. Everything for seed S goes under ../models/crime/seed_<S>/:
#   ../datasets/crime/crime_dataset
#     -> ../models/crime/seed_<S>/gemma3_4b_crime
#   ../datasets/crime/split_<N>/crime_dataset_<i>_of_<N>
#     -> ../models/crime/seed_<S>/split_<N>/gemma3_4b_crime_<i>_of_<N>
#   ../merge/crime/seed_<S>/crime_merge_<N>_<method>_config.yaml
#     -> ../models/crime/seed_<S>/split_<N>/gemma3_4b_crime_merge_<N>_<method>
# The existing unseeded models in ../models/crime/ are left alone.
#
# Usage, from train/:
#   bash submit_crime_seed.sh <S> [N...]            # dry run: checks paths, prints the chain
#   bash submit_crime_seed.sh <S> --submit [N...]   # submits, writing the job map
#
# Merge configs missing from ../merge/crime/seed_<S>/ are written first by
# ../merge/make_merge_configs.py crime <N> --seed <S> (also in a dry run).
#
# A step counts as complete when its output has a provenance.json (written
# only once the step succeeds) and its health check passed; complete steps are
# skipped, and the steps after them just don't wait for them, so a rerun picks
# up where the last one stopped. An output dir without one is left from a
# failed or running job: the script stops rather than overwrite it, so move it
# aside first. The job map (job, dependency, what) goes to ../evaluate/slurm/.
set -euo pipefail
cd "$(dirname "$0")"
USAGE="usage: bash submit_crime_seed.sh <seed> [--submit] [N...]"
SEED=${1:?$USAGE}; shift
[[ $SEED =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer, not $SEED" >&2; exit 1; }
SUBMIT=false
if [[ ${1:-} == --submit ]]; then SUBMIT=true; shift; fi
SPLITS=("${@:-2 4 8}")
SPLITS=(${SPLITS[*]})
CONFIG=crime_gemma.yaml
BASE=google/gemma-3-4b-it
MODELS=../models/crime/seed_$SEED
JOB_MAP=../evaluate/slurm/crime_seed_${SEED}_jobs_$(date +%F).txt
METHODS=(linear ties dare_ties task_arithmetic model_stock slerp arcee_fusion)
# Parallel jobs each running `uv sync` reinstall packages over one another.
export UV_NO_SYNC=1

# sub <dir> <dependency or -> <name> <description> <sbatch args...>: prints the job id.
sub() {
  local dir=$1 dep=$2 name=$3 what=$4; shift 4
  local args=(--parsable --job-name="$name")
  [[ $dep != - ]] && args+=(--dependency=afterok:$dep)
  local job
  if $SUBMIT; then
    job=$(cd "$dir" && sbatch "${args[@]}" "$@")
    echo "$job $dep $what" >> "$JOB_MAP"
  else
    job="<$name>"  # stands in for the job id in a dry run
    echo "  submit ($dir) sbatch ${args[*]} $*" >&2
  fi
  echo "$job"
}
need() { [[ -e $1 ]] || { echo "MISSING $1${2:+ ($2)}" >&2; exit 1; }; }
# complete <dir>: 0 if the step that writes dir has finished and its health
# check passed; stops the script if dir is half-made or unhealthy.
complete() {
  if [[ -f $1/provenance.json ]]; then
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$1/provenance.json'))['health']['ok'] else 1)" &&
      return 0
    echo "UNHEALTHY $1: its health check failed; move it aside to redo it" >&2; exit 1
  fi
  [[ -e $1 ]] && { echo "INCOMPLETE $1 has no provenance.json (failed or still running): move it aside" >&2; exit 1; }
  return 1
}
# train_and_eval <dataset> <output> <name>: the dependency for what needs the
# model (its training job, or - if it is already trained).
train_and_eval() {
  local data=$1 out=$2 name=$3 t
  if complete "$out"; then
    echo "  done   $out" >&2
    echo -
    return
  fi
  t=$(sub . - "$name" "train $out (seed $SEED)" train.sbatch "$CONFIG" "$data" "$out" --seed "$SEED")
  sub ../evaluate "$t" "eval_$name" "eval $out" evaluate.sbatch crime $BASE --adapter "$out" >/dev/null
  echo "$t"
}

# Check every input before submitting anything.
need ../datasets/crime/crime_dataset/train
for N in "${SPLITS[@]}"; do
  for ((i = 1; i <= N; i++)); do
    need ../datasets/crime/split_$N/crime_dataset_${i}_of_$N/train \
      "python ../split_dataset.py $N ../datasets/crime/crime_dataset ../datasets/crime/split_$N"
  done
  missing=false
  for m in "${METHODS[@]}"; do
    [[ -f ../merge/crime/seed_$SEED/crime_merge_${N}_${m}_config.yaml ]] || missing=true
  done
  if $missing; then
    (cd ../merge && python3 make_merge_configs.py crime "$N" --seed "$SEED" >/dev/null)
  fi
done

echo "== seed $SEED: full model" >&2
train_and_eval ../datasets/crime/crime_dataset "$MODELS/gemma3_4b_crime" "crime_s${SEED}" >/dev/null

for N in "${SPLITS[@]}"; do
  echo "== seed $SEED: N = $N" >&2
  DIR=$MODELS/split_$N
  TRAINING=()  # parts still to be trained, which the merges must wait for
  for ((i = 1; i <= N; i++)); do
    t=$(train_and_eval ../datasets/crime/split_$N/crime_dataset_${i}_of_$N \
      "$DIR/gemma3_4b_crime_${i}_of_$N" "crime_s${SEED}_${i}_of_$N")
    [[ $t != - ]] && TRAINING+=("$t")
  done
  PARTS=-
  ((${#TRAINING[@]})) && PARTS=$(IFS=:; echo "${TRAINING[*]}")
  # The merges share a LoRA merge cache, so the first one submitted fills it
  # before the others start.
  FIRST=
  for m in "${METHODS[@]}"; do
    OUT=$DIR/gemma3_4b_crime_merge_${N}_$m
    if complete "$OUT"; then
      echo "  done   $OUT" >&2
      continue
    fi
    j=$(sub ../merge "${FIRST:-$PARTS}" "merge_crime_s${SEED}_${N}_$m" "merge $OUT (seed $SEED)" \
      merge.sbatch crime/seed_$SEED/crime_merge_${N}_${m}_config.yaml "$OUT" \
      --seed "$SEED" --lora-merge-cache "$DIR/.lora_merge_cache")
    FIRST=${FIRST:-$j}
    sub ../evaluate "$j" "eval_crime_s${SEED}_merge_${N}_$m" "eval $OUT" \
      evaluate.sbatch crime "$OUT" >/dev/null
  done
done
if $SUBMIT; then echo "Submitted; job map: $JOB_MAP"; else echo "Dry run: nothing submitted. Rerun with --submit."; fi
