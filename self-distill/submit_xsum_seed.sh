#!/bin/bash
# Submits one repeat of the XSum experiments with seed S, as dependent chains:
#   full prompts    -> teacher -> KD student -> eval
#   each part of each N-way split
#                   -> teacher -> KD student -> eval each part
#                                            -> 7 merges of the N students -> eval each merge
# for each N given (default 2 4 8). As when the data is split over nodes, each
# part gets its own teacher run, and so does the full data, all in this one
# repeat. The teacher runs batch-invariant, without prefix caching or chunked
# prefill (see generate_teacher_logprobs.py), so its greedy outputs don't
# depend on how the prompts are split (99.98% identical), and every
# repeat should regenerate the same teacher data; it is regenerated anyway, as
# a node would. The prompts are the same for every seed (fixed prep seed,
# contiguous splits). Everything for seed S goes under seed_<S>/:
#   ../datasets/xsum/xsum_prompts
#     -> ../datasets/xsum/seed_<S>/gemma3_4b_xsum_kd_data_{short,long}
#     -> ../models/xsum/seed_<S>/gemma3_4b_xsum_short
#   ../datasets/xsum/split_<N>/xsum_prompts_<i>_of_<N>
#     -> ../datasets/xsum/seed_<S>/split_<N>/gemma3_4b_xsum_kd_data_<i>_of_<N>_{short,long}
#     -> ../models/xsum/seed_<S>/split_<N>/gemma3_4b_xsum_<i>_of_<N>_short
#   ../merge/xsum/seed_<S>/xsum_merge_<N>_<method>_config.yaml
#     -> ../models/xsum/seed_<S>/split_<N>/gemma3_4b_xsum_short_merge_<N>_<method>
# The teacher, training and merges all get --seed S (the teacher's only seeds
# vLLM, which greedy decoding doesn't use, but it is recorded). The existing
# unseeded data and models are left alone.
#
# Usage, from self-distill/:
#   bash submit_xsum_seed.sh <S> [N...]            # dry run: checks paths, prints the chain
#   bash submit_xsum_seed.sh <S> --submit [N...]   # submits, writing the job map
#
# Needs the KD configs self-distillation_xsum.yaml (full) and
# self-distillation_xsum_split_N.yaml. Merge configs missing from
# ../merge/xsum/seed_<S>/ are written first by
# ../merge/make_merge_configs.py xsum <N> --seed <S> (also in a dry run).
#
# A step counts as complete when its output has a provenance.json (written
# only once the step succeeds), and for a model only if its health check
# passed; complete steps are skipped, and the steps after them just don't wait
# for them, so a rerun picks up where the last one stopped. An output dir
# without one is left from a failed or running job: the script stops rather
# than overwrite it, so move it aside first. The job map (job, dependency,
# what) goes to ../evaluate/slurm/.
set -euo pipefail
cd "$(dirname "$0")"
USAGE="usage: bash submit_xsum_seed.sh <seed> [--submit] [N...]"
SEED=${1:?$USAGE}; shift
[[ $SEED =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer, not $SEED" >&2; exit 1; }
SUBMIT=false
if [[ ${1:-} == --submit ]]; then SUBMIT=true; shift; fi
SPLITS=("${@:-2 4 8}")
SPLITS=(${SPLITS[*]})
TEACHER=google/gemma-3-4b-it
DATA=../datasets/xsum/seed_$SEED
MODELS=../models/xsum/seed_$SEED
JOB_MAP=../evaluate/slurm/xsum_seed_${SEED}_jobs_$(date +%F).txt
METHODS=(linear ties dare_ties task_arithmetic model_stock slerp arcee_fusion)
# Parallel jobs each running `uv sync` reinstall packages over one another.
export UV_NO_SYNC=1

# A `uv sync` here undoes the axolotl KD NaN fix; refuse to train without it.
uv run python patch_axolotl_kd.py --check >/dev/null

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
# complete <dir> [model]: 0 if the step that writes dir has finished (and, for
# a model, its health check passed); stops the script if dir is half-made.
complete() {
  if [[ -f $1/provenance.json ]]; then
    [[ -z ${2:-} ]] && return 0
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$1/provenance.json'))['health']['ok'] else 1)" &&
      return 0
    echo "UNHEALTHY $1: its health check failed; move it aside to redo it" >&2; exit 1
  fi
  [[ -e $1 ]] && { echo "INCOMPLETE $1 has no provenance.json (failed or still running): move it aside" >&2; exit 1; }
  return 1
}
# teach_train_eval <prompts> <data dir> <prefix> <kd config> <student> <name>:
# the dependency for what needs the student (its training job, or - if it is
# already trained).
teach_train_eval() {
  local prompts=$1 dir=$2 prefix=$3 config=$4 student=$5 name=$6 dep=- t
  if complete "$student" model; then
    echo "  done   $student" >&2
    echo -
    return
  fi
  if complete "$dir/${prefix}_short"; then
    echo "  done   $dir/${prefix}_short" >&2
  else
    dep=$(sub . - "teach_$name" "teacher $prompts -> $dir/${prefix}_{short,long} (seed $SEED)" \
      generate_teacher_logprobs.sbatch $TEACHER "$prompts" "$dir" "$prefix" --seed "$SEED")
  fi
  t=$(sub . "$dep" "kd_$name" "train $student (seed $SEED)" \
    train_kd.sbatch "$config" "$student" "$dir/${prefix}_short" --seed "$SEED")
  sub ../evaluate "$t" "eval_$name" "eval $student short" \
    evaluate.sbatch xsum "$student" -T prompt_variant=short >/dev/null
  echo "$t"
}

# Check every input before submitting anything.
need self-distillation_xsum.yaml
need ../datasets/xsum/xsum_prompts/train
for N in "${SPLITS[@]}"; do
  need self-distillation_xsum_split_$N.yaml "scale the steps as in the existing split configs"
  for ((i = 1; i <= N; i++)); do
    need ../datasets/xsum/split_$N/xsum_prompts_${i}_of_$N/train \
      "python ../split_dataset.py $N ../datasets/xsum/xsum_prompts ../datasets/xsum/split_$N"
  done
  missing=false
  for m in "${METHODS[@]}"; do
    [[ -f ../merge/xsum/seed_$SEED/xsum_merge_${N}_${m}_config.yaml ]] || missing=true
  done
  if $missing; then
    (cd ../merge && python3 make_merge_configs.py xsum "$N" --seed "$SEED" >/dev/null)
  fi
done

echo "== seed $SEED: full data" >&2
teach_train_eval ../datasets/xsum/xsum_prompts "$DATA" gemma3_4b_xsum_kd_data \
  self-distillation_xsum.yaml "$MODELS/gemma3_4b_xsum_short" "xsum_s${SEED}" >/dev/null

for N in "${SPLITS[@]}"; do
  echo "== seed $SEED: N = $N" >&2
  DIR=$MODELS/split_$N
  TRAINING=()  # students still to be trained, which the merges must wait for
  for ((i = 1; i <= N; i++)); do
    t=$(teach_train_eval ../datasets/xsum/split_$N/xsum_prompts_${i}_of_$N "$DATA/split_$N" \
      gemma3_4b_xsum_kd_data_${i}_of_$N self-distillation_xsum_split_$N.yaml \
      "$DIR/gemma3_4b_xsum_${i}_of_${N}_short" "xsum_s${SEED}_${i}_of_$N")
    [[ $t != - ]] && TRAINING+=("$t")
  done
  ALL=-
  ((${#TRAINING[@]})) && ALL=$(IFS=:; echo "${TRAINING[*]}")
  for m in "${METHODS[@]}"; do
    OUT=$DIR/gemma3_4b_xsum_short_merge_${N}_$m
    if complete "$OUT" model; then
      echo "  done   $OUT" >&2
      continue
    fi
    j=$(sub ../merge "$ALL" "merge_xsum_s${SEED}_${N}_$m" "merge $OUT (seed $SEED)" \
      merge.sbatch xsum/seed_$SEED/xsum_merge_${N}_${m}_config.yaml "$OUT" --seed "$SEED")
    sub ../evaluate "$j" "eval_xsum_s${SEED}_merge_${N}_$m" "eval $OUT short" \
      evaluate.sbatch xsum "$OUT" -T prompt_variant=short >/dev/null
  done
done
if $SUBMIT; then echo "Submitted; job map: $JOB_MAP"; else echo "Dry run: nothing submitted. Rerun with --submit."; fi
