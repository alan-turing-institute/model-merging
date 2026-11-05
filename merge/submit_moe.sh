#!/bin/bash
# Submits the task MoE of one model and seed, followed by its evaluation on all
# four tasks (CLINC150 with the labels prompt, XSum with the long one), as for
# the dense task merges (submit_task_merges.sh): 5 jobs.
#
# Usage, from merge/:
#   bash submit_moe.sh <mistral|llama> <seed> [--submit]
#
# The MoE waits for the jobs that make its experts, found by name: the three
# conversions (convert_<task>_<model>_s<S>) and the XSum KD run
# (mlp_xsum_<model>_s<S>). One still queued or running becomes a dependency;
# one already finished must have COMPLETED, and its output must exist, or the
# script stops. Without --submit it prints the sbatch commands only.
set -euo pipefail
cd "$(dirname "$0")"
USAGE="usage: bash submit_moe.sh <mistral|llama> <seed> [--submit]"
MODEL=${1:?$USAGE}; SEED=${2:?$USAGE}; SUBMIT=false
[[ ${3:-} == --submit ]] && SUBMIT=true
export UV_NO_SYNC=1

read -r CONFIG OUT < <(python3 make_moe_configs.py "$MODEL" --seed "$SEED" 2>/dev/null)
if [[ -f $OUT/provenance.json ]]; then echo "  done   $OUT" >&2; exit 0; fi
[[ -e $OUT ]] && { echo "INCOMPLETE $OUT has no provenance.json: move it aside" >&2; exit 1; }

DEPS=()
for name in convert_crime_${MODEL}_s$SEED convert_wine_${MODEL}_s$SEED \
            convert_clinc150_${MODEL}_s$SEED mlp_xsum_${MODEL}_s$SEED; do
  job=$(squeue --me -h -n "$name" -o %i | head -1)
  if [[ -n $job ]]; then DEPS+=("$job"); continue; fi
  state=$(sacct -u "$USER" -S 2026-11-03 -X -n -P --name "$name" --format=State | tail -1)
  [[ $state == COMPLETED ]] || { echo "$name: not queued and last state '${state:-none}', not COMPLETED" >&2; exit 1; }
done
# ...and every expert dir the config names must be there once those jobs finish;
# for the finished ones, it must be there now, with a provenance.json.
python3 - "$CONFIG" "${#DEPS[@]}" <<'EOF'
import sys, yaml, os
experts = [e["source_model"] for e in yaml.safe_load(open(sys.argv[1]))["experts"]]
missing = [e for e in experts if not os.path.exists(os.path.join(e, "provenance.json"))]
if missing and sys.argv[2] == "0":
    sys.exit(f"experts not finished and nothing to wait for: {missing}")
EOF
DEP=-; ((${#DEPS[@]})) && DEP=$(IFS=:; echo "${DEPS[*]}")

JOB_MAP=../evaluate/slurm/task_moe_${MODEL}_seed_${SEED}_jobs_$(date +%F).txt
# XSum at batch 2, not evaluate.sbatch's 8 (a later -M wins): ~48 GB of MoE
# weights plus 8 long articles ran out of GPU memory (3 Nov). Decoding is greedy,
# so only the padding differs.
TASKS=("crime:" "wine:" "clinc150:-T prompt_variant=labels" "xsum:-T prompt_variant=long -M batch_size=2")
sub() {  # sub <dir> <dependency or -> <name> <offline?> <sbatch args...>: prints the job id.
  local dir=$1 dep=$2 name=$3 offline=$4; shift 4
  local args=(--parsable --job-name="$name")
  [[ $dep != - ]] && args+=(--dependency=afterok:$dep)
  if $SUBMIT; then
    local job
    if [[ $offline == 1 ]]; then job=$(cd "$dir" && HF_HUB_OFFLINE=1 sbatch "${args[@]}" "$@")
    else job=$(cd "$dir" && env -u HF_HUB_OFFLINE sbatch "${args[@]}" "$@"); fi
    echo "$job $dep $name" >> "$JOB_MAP"; echo "$job"
  else
    echo "  submit ($dir) sbatch ${args[*]} $*" >&2; echo "<$name>"
  fi
}
m=$(sub . "$DEP" "moe_${MODEL}_s$SEED" 0 moe.sbatch "$CONFIG" "$OUT" --seed "$SEED")
for t in "${TASKS[@]}"; do
  task=${t%%:*}; extra=${t#*:}
  # shellcheck disable=SC2086
  # A 4-expert MoE (~24B parameters) takes longer to evaluate than a dense model.
  # XSum also gets expandable segments, against fragmentation from its long
  # prompts; sbatch passes the variable on to the job.
  if [[ $task == xsum ]]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      sub ../evaluate "$m" "eval_moe_${task}_${MODEL}_s$SEED" 1 --time=03:00:00 evaluate.sbatch "$task" "$OUT" $extra >/dev/null
  else
    sub ../evaluate "$m" "eval_moe_${task}_${MODEL}_s$SEED" 1 --time=03:00:00 evaluate.sbatch "$task" "$OUT" $extra >/dev/null
  fi
done
if $SUBMIT; then echo "Submitted; job map: $JOB_MAP"; else echo "Dry run: nothing submitted. Rerun with --submit."; fi
