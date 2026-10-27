#!/bin/bash
# Submits the XSum data-parallel split experiments as dependent chains, for
# each N given (default 2 4 8):
#   teacher on each part's prompts -> KD student -> eval each part
#                                  -> 7 merges of the N students -> eval each merge
# Any step whose output is already complete is skipped, and the steps after it
# just don't wait for it. So a rerun picks up where the last one stopped, and a
# KD dataset that already exists is reused rather than regenerated.
#
# Usage, from self-distill/:
#   bash submit_xsum_splits.sh [N...]            # dry run: checks paths, prints the chain
#   bash submit_xsum_splits.sh --submit [N...]   # submits, writing the job map
#
# Needs, per N:
#   ../datasets/xsum/split_N/xsum_prompts_<i>_of_N    from ../split_dataset.py N
#   self-distillation_xsum_split_N.yaml               the KD config, with warmup,
#       eval and save steps scaled to one part's steps (see the existing ones)
#   ../merge/xsum/xsum_merge_N_<method>_config.yaml   from ../merge/make_merge_configs.py xsum N
# Writes, per part i of N:
#   ../datasets/xsum/split_N/gemma3_4b_xsum_kd_data_<i>_of_N_{short,long}
#   ../models/xsum/split_N/gemma3_4b_xsum_<i>_of_N_short
# and the merges ../models/xsum/split_N/gemma3_4b_xsum_short_merge_N_<method>.
#
# A step counts as complete when its output has a provenance.json (written
# only once the step succeeds), and for a model only if its health check
# passed. An output dir without one is left from a failed or running job: the
# script stops rather than overwrite it, so move it aside first.
# The job map (job, dependency, what) goes to ../evaluate/slurm/.
set -euo pipefail
cd "$(dirname "$0")"
SUBMIT=false
if [[ ${1:-} == --submit ]]; then SUBMIT=true; shift; fi
SPLITS=("${@:-2 4 8}")
SPLITS=(${SPLITS[*]})
JOB_MAP=../evaluate/slurm/xsum_split_jobs_$(date +%F).txt
METHODS=(linear ties dare_ties task_arithmetic model_stock slerp arcee_fusion)
TEACHER=google/gemma-3-4b-it
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

# Check every input before submitting anything.
for N in "${SPLITS[@]}"; do
  need self-distillation_xsum_split_$N.yaml "scale the steps as in the existing split configs"
  for m in "${METHODS[@]}"; do
    need ../merge/xsum/xsum_merge_${N}_${m}_config.yaml "python ../merge/make_merge_configs.py xsum $N"
  done
  for ((i = 1; i <= N; i++)); do
    complete ../datasets/xsum/split_$N/gemma3_4b_xsum_kd_data_${i}_of_${N}_short ||
      need ../datasets/xsum/split_$N/xsum_prompts_${i}_of_$N "python ../split_dataset.py $N ../datasets/xsum/xsum_prompts ../datasets/xsum/split_$N"
  done
done

for N in "${SPLITS[@]}"; do
  echo "== N = $N" >&2
  TRAINING=()  # students still to be trained, which the merges must wait for
  for ((i = 1; i <= N; i++)); do
    DATA=../datasets/xsum/split_$N/gemma3_4b_xsum_kd_data_${i}_of_$N
    STUDENT=../models/xsum/split_$N/gemma3_4b_xsum_${i}_of_${N}_short
    if complete "$STUDENT" model; then
      echo "  done   $STUDENT" >&2
      continue
    fi
    dep=-
    if complete "${DATA}_short"; then
      echo "  done   ${DATA}_short" >&2
    else
      dep=$(sub . - "teach_${i}_of_$N" "teacher xsum $i of $N -> ${DATA}_{short,long}" \
        generate_teacher_logprobs.sbatch $TEACHER ../datasets/xsum/split_$N/xsum_prompts_${i}_of_$N \
        ../datasets/xsum/split_$N gemma3_4b_xsum_kd_data_${i}_of_$N)
    fi
    t=$(sub . "$dep" "kd_${i}_of_$N" "train $STUDENT" \
      train_kd.sbatch self-distillation_xsum_split_$N.yaml "$STUDENT" "${DATA}_short")
    TRAINING+=("$t")
    sub ../evaluate "$t" "eval_xsum_${i}_of_$N" "eval $STUDENT short" \
      evaluate.sbatch xsum "$STUDENT" -T prompt_variant=short >/dev/null
  done
  ALL=-
  ((${#TRAINING[@]})) && ALL=$(IFS=:; echo "${TRAINING[*]}")
  for m in "${METHODS[@]}"; do
    OUT=../models/xsum/split_$N/gemma3_4b_xsum_short_merge_${N}_$m
    if complete "$OUT" model; then
      echo "  done   $OUT" >&2
      continue
    fi
    j=$(sub ../merge "$ALL" "merge_xsum_${N}_$m" "merge $OUT" \
      merge.sbatch xsum/xsum_merge_${N}_${m}_config.yaml "$OUT")
    sub ../evaluate "$j" "eval_xsum_merge_${N}_$m" "eval $OUT short" \
      evaluate.sbatch xsum "$OUT" -T prompt_variant=short >/dev/null
  done
done
if $SUBMIT; then echo "Submitted; job map: $JOB_MAP"; else echo "Dry run: nothing submitted. Rerun with --submit."; fi
