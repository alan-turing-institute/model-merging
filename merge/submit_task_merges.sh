#!/bin/bash
# Submits the task merges of one model and seed, each followed by its
# evaluation on all four tasks:
#   merge (7 methods, configs from make_task_merge_configs.py)
#     -> eval crime, wine, clinc150 (-T prompt_variant=labels), xsum (-T prompt_variant=long)
# The merged models go to ../models/task_merge/seed_<S>/.
#
# Usage, from merge/:
#   bash submit_task_merges.sh <gemma|mistral|llama> <seed> [--after <job:job...>] [--submit]
# --after holds the merges until those jobs (the experts' training) have
# succeeded. Of those, only the ones still queued or running become
# dependencies: Slurm refuses a dependency on a job it has already forgotten
# ("Job dependency problem"), so one that has finished must have COMPLETED, or
# the script stops. Without --submit it prints the sbatch commands only.
#
# The three LoRA experts are merged into their base once, into a cache that all
# the methods share, so the first merge (linear) runs alone and the others wait
# for it. Merges run online (HF_HUB_OFFLINE unset): mergekit lists the base
# model's files on the Hub, which offline mode refuses. Evaluations run offline.
set -euo pipefail
cd "$(dirname "$0")"
USAGE="usage: bash submit_task_merges.sh <gemma|mistral|llama> <seed> [--after <job:job...>] [--submit]"
MODEL=${1:?$USAGE}; SEED=${2:?$USAGE}; shift 2
AFTER=; SUBMIT=false
while (($#)); do
  case $1 in
    --after) AFTER=${2:?$USAGE}; shift ;;
    --submit) SUBMIT=true ;;
    *) echo "$USAGE" >&2; exit 1 ;;
  esac
  shift
done
case $MODEL in
  gemma) BASE=google/gemma-3-4b-it ;;
  mistral) BASE=mistralai/Mistral-7B-Instruct-v0.3 ;;
  llama) BASE=meta-llama/Llama-3.1-8B-Instruct ;;
  *) echo "$USAGE" >&2; exit 1 ;;
esac
export UV_NO_SYNC=1
# Seeds listed in SKIP_SEEDS (one per line) are skipped: a budget cut, 5 Nov
# (task merging regenerated for seeds 1-3 only). Delete the file to stop skipping.
if [[ -f SKIP_SEEDS ]] && grep -qx "$SEED" SKIP_SEEDS; then
  echo "  skip   seed $SEED (listed in merge/SKIP_SEEDS)" >&2; exit 0
fi
if [[ -n $AFTER ]]; then
  WAITING=()
  for job in ${AFTER//:/ }; do
    if [[ -n $(squeue -h -j "$job" -o %i 2>/dev/null) ]]; then WAITING+=("$job"); continue; fi
    state=$(sacct -j "$job" -X -n -P --format=State | head -1)
    [[ $state == COMPLETED ]] || { echo "--after job $job: not queued and state '${state:-unknown}', not COMPLETED" >&2; exit 1; }
  done
  AFTER=$(IFS=:; echo "${WAITING[*]}")
fi
CACHE=../models/task_merge/seed_$SEED/.lora_merge_cache_$MODEL
JOB_MAP=../evaluate/slurm/task_merge_${MODEL}_seed_${SEED}_jobs_$(date +%F).txt
TASKS=("crime:" "wine:" "clinc150:-T prompt_variant=labels" "xsum:-T prompt_variant=long")

# sub <dir> <dependency or -> <name> <offline?> <sbatch args...>: prints the job id.
sub() {
  local dir=$1 dep=$2 name=$3 offline=$4; shift 4
  local args=(--parsable --job-name="$name")
  [[ $dep != - ]] && args+=(--dependency=afterok:$dep)
  if $SUBMIT; then
    local job
    if [[ $offline == 1 ]]; then
      job=$(cd "$dir" && HF_HUB_OFFLINE=1 sbatch "${args[@]}" "$@")
    else
      job=$(cd "$dir" && env -u HF_HUB_OFFLINE sbatch "${args[@]}" "$@")
    fi
    echo "$job $dep $name" >> "$JOB_MAP"
    echo "$job"
  else
    echo "  submit ($dir) sbatch ${args[*]} $*" >&2
    echo "<$name>"
  fi
}

FIRST=
while read -r METHOD CONFIG OUT; do
  if [[ -f $OUT/provenance.json ]]; then echo "  done   $OUT" >&2; continue; fi
  [[ -e $OUT ]] && { echo "INCOMPLETE $OUT has no provenance.json: move it aside" >&2; exit 1; }
  # Pairwise trees run three mergekit stages, and arcee_fusion is slow (up to
  # 23 min on 3 Nov): allow 1.5 h.
  m=$(sub . "${FIRST:-${AFTER:--}}" "tm_${MODEL}_${METHOD}_s$SEED" 0 --time=01:30:00 \
        merge.sbatch "$CONFIG" "$OUT" --seed "$SEED" --lora-merge-cache "$CACHE")
  FIRST=${FIRST:-$m}
  for t in "${TASKS[@]}"; do
    task=${t%%:*}; extra=${t#*:}
    # shellcheck disable=SC2086  # extra is "" or "-T prompt_variant=..."
    sub ../evaluate "$m" "eval_tm_${task}_${MODEL}_${METHOD}_s$SEED" 1 evaluate.sbatch "$task" "$OUT" $extra >/dev/null
  done
done < <(python3 make_task_merge_configs.py "$MODEL" --seed "$SEED")
if $SUBMIT; then echo "Submitted; job map: $JOB_MAP"; else echo "Dry run: nothing submitted. Rerun with --submit."; fi
