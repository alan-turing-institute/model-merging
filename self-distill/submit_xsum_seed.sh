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
# contiguous splits).
#
# --targets gold (the default) trains on XSum's gold summaries, which the
# teacher only scores; --targets teacher trains on the teacher's own greedy
# summaries. Gold runs have _gold in every name (<G> below), so the two
# don't collide. --prompt long (the default) trains and evaluates the students
# with the long student prompt, which gives the base model enough instruction
# for a fair comparison; --prompt short uses the short one (<P> below; _long
# in the merge task's name <T> for long). The teacher writes both prompts'
# data in one run, so long and short runs share it. Everything for seed S
# goes under seed_<S>/:
#   ../datasets/xsum/xsum_prompts
#     -> ../datasets/xsum/seed_<S>/gemma3_4b_xsum<G>_kd_data_{short,long}
#     -> ../models/xsum/seed_<S>/gemma3_4b_xsum<G>_<P>
#   ../datasets/xsum/split_<N>/xsum_prompts_<i>_of_<N>
#     -> ../datasets/xsum/seed_<S>/split_<N>/gemma3_4b_xsum<G>_kd_data_<i>_of_<N>_{short,long}
#     -> ../models/xsum/seed_<S>/split_<N>/gemma3_4b_xsum<G>_<i>_of_<N>_<P>
#   ../merge/xsum/seed_<S>/<T>_merge_<N>_<method>_config.yaml
#     -> ../models/xsum/seed_<S>/split_<N>/gemma3_4b_xsum<G>_<P>_merge_<N>_<method>
# The teacher, training and merges all get --seed S (the teacher's only seeds
# vLLM, which greedy decoding doesn't use, but it is recorded). The existing
# unseeded data and models are left alone.
#
# --base B (default google/gemma-3-4b-it) is both the teacher and the students'
# starting point, as self-distillation needs. For any other base, the teacher
# data, models and merge configs go under xsum_<size> instead of xsum (e.g.
# ../models/xsum_1b/seed_<S>/), and gemma3_4b in every name above becomes
# gemma3_<size>; the prompts are shared.
#
# Usage, from self-distill/:
#   bash submit_xsum_seed.sh <S> [N...]            # dry run: checks paths, prints the chain
#   bash submit_xsum_seed.sh <S> --submit [N...]   # submits, writing the job map
#   bash submit_xsum_seed.sh <S> --targets teacher --prompt short --submit [N...]
#   bash submit_xsum_seed.sh <S> --base google/gemma-3-1b-it --submit [N...]
#
# Needs the KD configs self-distillation_xsum.yaml (full) and
# self-distillation_xsum_split_N.yaml. Merge configs missing from
# ../merge/xsum/seed_<S>/ are written first by
# ../merge/make_merge_configs.py <T> <N> --seed <S> (also in a dry run).
#
# A step counts as complete when its output has a provenance.json (written
# only once the step succeeds), and for a model only if its health check
# passed; complete steps are skipped, and the steps after them just don't wait
# for them, so a rerun picks up where the last one stopped. An output dir
# without one is left from a failed or running job: the script stops rather
# than overwrite it, so move it aside first. The job map (job, dependency,
# what) goes to ../evaluate/slurm/<T>_seed_<S>_jobs_<date>.txt.
set -euo pipefail
cd "$(dirname "$0")"
USAGE="usage: bash submit_xsum_seed.sh <seed> [--targets gold|teacher] [--prompt long|short] [--base <model>] [--submit] [N...]"
SEED=${1:?$USAGE}; shift
[[ $SEED =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer, not $SEED" >&2; exit 1; }
SUBMIT=false
TARGETS=gold
PROMPT=long
BASE=google/gemma-3-4b-it
SPLITS=()
while (($#)); do
  case $1 in
    --submit) SUBMIT=true ;;
    --targets) TARGETS=${2:?$USAGE}; shift ;;
    --targets=*) TARGETS=${1#--targets=} ;;
    --prompt) PROMPT=${2:?$USAGE}; shift ;;
    --prompt=*) PROMPT=${1#--prompt=} ;;
    --base) BASE=${2:?$USAGE}; shift ;;
    --base=*) BASE=${1#--base=} ;;
    -*) echo "$USAGE" >&2; exit 1 ;;
    *) SPLITS+=("$1") ;;
  esac
  shift
done
((${#SPLITS[@]})) || SPLITS=(2 4 8)
case $TARGETS in
  gold) G=_gold ;;
  teacher) G= ;;
  *) echo "--targets must be gold or teacher, not $TARGETS" >&2; exit 1 ;;
esac
case $PROMPT in
  long) P=_long ;;
  short) P= ;;
  *) echo "--prompt must be long or short, not $PROMPT" >&2; exit 1 ;;
esac
[[ $BASE =~ ^google/gemma-3-([0-9]+b)-it$ ]] || { echo "--base must be google/gemma-3-<size>-it, not $BASE" >&2; exit 1; }
SIZE=${BASH_REMATCH[1]}
X=xsum  # the folder of everything made from this base
[[ $BASE == google/gemma-3-4b-it ]] || X=xsum_$SIZE
TEACHER=$BASE
DATA=../datasets/$X/seed_$SEED
MODELS=../models/$X/seed_$SEED
MERGE_TASK=xsum$G$P  # make_merge_configs.py's task
NAME=gemma3_${SIZE}_xsum$G
JOB=${MERGE_TASK}${X#xsum}_s$SEED  # job name stem
mkdir -p ../evaluate/slurm
JOB_MAP=../evaluate/slurm/${MERGE_TASK}${X#xsum}_seed_${SEED}_jobs_$(date +%F).txt
METHODS=(linear ties dare_ties task_arithmetic model_stock slerp arcee_fusion)
# Parallel jobs each running `uv sync` reinstall packages over one another.
export UV_NO_SYNC=1

# A `uv sync` here undoes the axolotl KD NaN fix; refuse to train without it.
uv run python patch_axolotl_kd.py --check >/dev/null

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
  if complete "$dir/${prefix}_$PROMPT"; then
    echo "  done   $dir/${prefix}_$PROMPT" >&2
  else
    dep=$(sub . - "teach_$name" "teacher ($TARGETS) $prompts -> $dir/${prefix}_{short,long} (seed $SEED)" \
      generate_teacher_logprobs.sbatch $TEACHER "$prompts" "$dir" "$prefix" --targets "$TARGETS" --seed "$SEED")
  fi
  t=$(sub . "$dep" "kd_$name" "train $student (seed $SEED)" \
    train_kd.sbatch "$config" "$student" "$dir/${prefix}_$PROMPT" --seed "$SEED" --base-model "$BASE")
  sub ../evaluate "$t" "eval_$name" "eval $student $PROMPT" \
    evaluate.sbatch xsum "$student" -T prompt_variant=$PROMPT >/dev/null
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
    [[ -f ../merge/$X/seed_$SEED/${MERGE_TASK}_merge_${N}_${m}_config.yaml ]] || missing=true
  done
  if $missing; then
    (cd ../merge && python3 make_merge_configs.py "$MERGE_TASK" "$N" --seed "$SEED" --base "$BASE" >/dev/null)
  fi
done

echo "== seed $SEED ($TARGETS targets, $PROMPT prompt): full data" >&2
teach_train_eval ../datasets/xsum/xsum_prompts "$DATA" ${NAME}_kd_data \
  self-distillation_xsum.yaml "$MODELS/${NAME}_$PROMPT" "$JOB" >/dev/null

for N in "${SPLITS[@]}"; do
  echo "== seed $SEED ($TARGETS targets, $PROMPT prompt): N = $N" >&2
  DIR=$MODELS/split_$N
  TRAINING=()  # students still to be trained, which the merges must wait for
  for ((i = 1; i <= N; i++)); do
    t=$(teach_train_eval ../datasets/xsum/split_$N/xsum_prompts_${i}_of_$N "$DATA/split_$N" \
      ${NAME}_kd_data_${i}_of_$N self-distillation_xsum_split_$N.yaml \
      "$DIR/${NAME}_${i}_of_${N}_$PROMPT" "${JOB}_${i}_of_$N")
    [[ $t != - ]] && TRAINING+=("$t")
  done
  ALL=-
  ((${#TRAINING[@]})) && ALL=$(IFS=:; echo "${TRAINING[*]}")
  for m in "${METHODS[@]}"; do
    OUT=$DIR/${NAME}_${PROMPT}_merge_${N}_$m
    if complete "$OUT" model; then
      echo "  done   $OUT" >&2
      continue
    fi
    j=$(sub ../merge "$ALL" "merge_${JOB}_${N}_$m" "merge $OUT (seed $SEED)" \
      merge.sbatch $X/seed_$SEED/${MERGE_TASK}_merge_${N}_${m}_config.yaml "$OUT" --seed "$SEED")
    sub ../evaluate "$j" "eval_${JOB}_merge_${N}_$m" "eval $OUT $PROMPT" \
      evaluate.sbatch xsum "$OUT" -T prompt_variant=$PROMPT >/dev/null
  done
done
if $SUBMIT; then echo "Submitted; job map: $JOB_MAP"; else echo "Dry run: nothing submitted. Rerun with --submit."; fi
