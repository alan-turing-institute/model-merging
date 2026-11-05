#!/bin/bash
# Submits one repeat of a LoRA task's data-parallel experiments (crime, wine or
# mt_domain_de_en), trained and merged with seed S from base model B, as
# submit_crime_seed.sh does for crime at 4B, as dependent chains:
#   full model                    -> eval
#   each part of each N-way split -> eval each part
#                                 -> 7 merges of the N parts -> eval each merge
# for each N given (default 2 4 8). The data is the same for every seed and
# size (the prep seed and the contiguous splits are fixed), so repeats differ
# only in training (data order, dropout, LoRA initialisation) and in dare_ties'
# random dropping. Everything for seed S goes under ../models/<T>/seed_<S>/,
# where <T> is the task, with _<size> after it for any base but gemma-3-4b-it
# (crime_1b for google/gemma-3-1b-it), and <M> is gemma3_<size>_<task>:
#   ../datasets/<task>/<task>_dataset
#     -> ../models/<T>/seed_<S>/<M>
#   ../datasets/<task>/split_<N>/<task>_dataset_<i>_of_<N>
#     -> ../models/<T>/seed_<S>/split_<N>/<M>_<i>_of_<N>
#   ../merge/<T>/seed_<S>/<task>_merge_<N>_<method>_config.yaml
#     -> ../models/<T>/seed_<S>/split_<N>/<M>_merge_<N>_<method>
# Every model is trained with <task>_gemma.yaml, with its base_model replaced
# by B.
#
# Usage, from train/:
#   bash submit_lora_seed.sh <task> <S> [--base <model>] [N...]            # dry run
#   bash submit_lora_seed.sh <task> <S> [--base <model>] --submit [N...]   # submits, writing the job map
#   e.g. bash submit_lora_seed.sh wine 1 --base google/gemma-3-1b-it --submit
#
# Merge configs missing from ../merge/<T>/seed_<S>/ are written first by
# ../merge/make_merge_configs.py <task> <N> --seed <S> --base <B> (also in a dry
# run).
#
# A step counts as complete when its output has a provenance.json (written
# only once the step succeeds) and its health check passed; complete steps are
# skipped, and the steps after them just don't wait for them, so a rerun picks
# up where the last one stopped. An output dir without one is left from a
# failed or running job: the script stops rather than overwrite it, so move it
# aside first. The job map (job, dependency, what) goes to ../evaluate/slurm/.
set -euo pipefail
cd "$(dirname "$0")"
USAGE="usage: bash submit_lora_seed.sh <crime|wine|mt_domain_de_en> <seed> [--base <model>] [--submit] [N...]"
TASK=${1:?$USAGE}
SEED=${2:?$USAGE}
shift 2
[[ $SEED =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer, not $SEED" >&2; exit 1; }
SUBMIT=false
BASE=google/gemma-3-4b-it
SPLITS=()
while (($#)); do
  case $1 in
    --submit) SUBMIT=true ;;
    --base) BASE=${2:?$USAGE}; shift ;;
    -*) echo "$USAGE" >&2; exit 1 ;;
    *) SPLITS+=("$1") ;;
  esac
  shift
done
((${#SPLITS[@]})) || SPLITS=(2 4 8)
[[ $BASE =~ ^google/gemma-3-([0-9]+b)-it$ ]] || { echo "--base must be google/gemma-3-<size>-it, not $BASE" >&2; exit 1; }
SIZE=${BASH_REMATCH[1]}
case $TASK in
  crime | wine) EVAL_TIME=01:00:00 ;;
  # Generation runs to a sentence or two rather than one word.
  mt_domain_de_en) EVAL_TIME=02:00:00 ;;
  *) echo "$USAGE" >&2; exit 1 ;;
esac
T=$TASK
[[ $BASE == google/gemma-3-4b-it ]] || T=${TASK}_$SIZE
CONFIG=${TASK}_gemma.yaml
NAME=gemma3_${SIZE}_$TASK
DATA=../datasets/$TASK
MODELS=../models/$T/seed_$SEED
mkdir -p ../evaluate/slurm
JOB_MAP=../evaluate/slurm/${T}_seed_${SEED}_jobs_$(date +%F).txt
METHODS=(linear ties dare_ties task_arithmetic model_stock slerp arcee_fusion)
# Parallel jobs each running `uv sync` reinstall packages over one another.
export UV_NO_SYNC=1

# sub <dir> <dependency or -> <name> <description> <sbatch args...>: prints the job id.
sub() {
  local dir=$1 dep=$2 name=$3 what=$4; shift 4
  local args=(--parsable --job-name="$name")
  # A failed step cancels the steps that wait for it, rather than leaving them
  # pending for ever.
  [[ $dep != - ]] && args+=(--dependency=afterok:$dep --kill-on-invalid-dep=yes)
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
  t=$(sub . - "$name" "train $out (seed $SEED)" \
    train.sbatch "$CONFIG" "$data" "$out" --seed "$SEED" --base-model "$BASE")
  sub ../evaluate "$t" "eval_$name" "eval $out" \
    --time=$EVAL_TIME evaluate.sbatch "$TASK" "$BASE" --adapter "$out" >/dev/null
  echo "$t"
}

# Check every input before submitting anything.
need "$CONFIG"
need $DATA/${TASK}_dataset/train
for N in "${SPLITS[@]}"; do
  for ((i = 1; i <= N; i++)); do
    need $DATA/split_$N/${TASK}_dataset_${i}_of_$N/train \
      "python ../split_dataset.py $N $DATA/${TASK}_dataset $DATA/split_$N"
  done
  missing=false
  for m in "${METHODS[@]}"; do
    [[ -f ../merge/$T/seed_$SEED/${TASK}_merge_${N}_${m}_config.yaml ]] || missing=true
  done
  if $missing; then
    (cd ../merge && python3 make_merge_configs.py "$TASK" "$N" --seed "$SEED" --base "$BASE" >/dev/null)
  fi
done

echo "== $T seed $SEED: full model" >&2
train_and_eval $DATA/${TASK}_dataset "$MODELS/$NAME" "${T}_s${SEED}" >/dev/null

for N in "${SPLITS[@]}"; do
  echo "== $T seed $SEED: N = $N" >&2
  DIR=$MODELS/split_$N
  TRAINING=()  # parts still to be trained, which the merges must wait for
  for ((i = 1; i <= N; i++)); do
    t=$(train_and_eval $DATA/split_$N/${TASK}_dataset_${i}_of_$N \
      "$DIR/${NAME}_${i}_of_$N" "${T}_s${SEED}_${i}_of_$N")
    [[ $t != - ]] && TRAINING+=("$t")
  done
  PARTS=-
  ((${#TRAINING[@]})) && PARTS=$(IFS=:; echo "${TRAINING[*]}")
  # The merges share a LoRA merge cache, so the first one submitted fills it
  # before the others start.
  FIRST=
  for m in "${METHODS[@]}"; do
    OUT=$DIR/${NAME}_merge_${N}_$m
    if complete "$OUT"; then
      echo "  done   $OUT" >&2
      continue
    fi
    j=$(sub ../merge "${FIRST:-$PARTS}" "merge_${T}_s${SEED}_${N}_$m" "merge $OUT (seed $SEED)" \
      merge.sbatch $T/seed_$SEED/${TASK}_merge_${N}_${m}_config.yaml "$OUT" \
      --seed "$SEED" --lora-merge-cache "$DIR/.lora_merge_cache")
    FIRST=${FIRST:-$j}
    sub ../evaluate "$j" "eval_${T}_s${SEED}_merge_${N}_$m" "eval $OUT" \
      --time=$EVAL_TIME evaluate.sbatch "$TASK" "$OUT" >/dev/null
  done
done
if $SUBMIT; then echo "Submitted; job map: $JOB_MAP"; else echo "Dry run: nothing submitted. Rerun with --submit."; fi
