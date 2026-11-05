#!/bin/bash
#SBATCH --job-name=dp-queue
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=evaluate/slurm/%x-%j.out
#SBATCH --error=evaluate/slurm/%x-%j.out
# Submits the data-parallel repeats of every task from one base model, seed by
# seed: crime, wine and German->English translation through
# train/submit_lora_seed.sh, and XSum (gold targets, long prompt) through
# self-distill/submit_xsum_seed.sh, after one evaluation of the untrained base
# model per task. Isambard-AI's workq_qos takes at most 512 queued jobs per
# user, and one repeat of one task is 72 jobs (87 for XSum, with its teachers),
# so the steps go in as the queue drains: each run submits the steps that fit,
# in order, and if any are left resubmits itself to run again in 10 minutes.
# It runs as a job because the login nodes end a session's processes when it
# closes.
#
# Usage, from the repo root:
#   sbatch submit_data_parallel_queue.sh <base model> [seed...]
#   e.g. sbatch submit_data_parallel_queue.sh google/gemma-3-1b-it 1 2 3 4 5
#
# Each submitted step is appended to evaluate/slurm/queue_<size>_done.txt and
# skipped by later runs, so after a failure the chain can simply be submitted
# again. The launchers' own job maps say which jobs each step submitted.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
USAGE="usage: sbatch submit_data_parallel_queue.sh <base model> [seed...]"
BASE=${1:?$USAGE}
shift
SEEDS=("${@:-1 2 3 4 5}")
SEEDS=(${SEEDS[*]})
[[ $BASE =~ ^google/gemma-3-([0-9]+b)-it$ ]] || { echo "base must be google/gemma-3-<size>-it, not $BASE" >&2; exit 1; }
SIZE=${BASH_REMATCH[1]}
LIMIT=500  # of workq_qos's 512, leaving room for jobs submitted by hand
mkdir -p evaluate/slurm
DONE=evaluate/slurm/queue_${SIZE}_done.txt
touch "$DONE"
export UV_NO_SYNC=1

# step <name> <jobs> <command...>: runs the command unless an earlier run did;
# fails, without running it, if the queue has no room for its jobs.
step() {
  local name=$1 jobs=$2; shift 2
  grep -qxF "$name" "$DONE" && return 0
  (( $(squeue -u "$USER" -h -r | wc -l) + jobs <= LIMIT )) || return 1
  echo "$(date -u +%FT%TZ) submit $name"
  # set -e is off inside `if all`, so a failed launcher has to stop the chain
  # itself (it checks its inputs before submitting anything).
  "$@" || { echo "$(date -u +%FT%TZ) FAILED $name: fix it and submit the chain again" >&2; exit 1; }
  echo "$name" >> "$DONE"
}
base_evals() {
  (cd evaluate &&
    sbatch --parsable --job-name="eval_${SIZE}_base_crime" evaluate.sbatch crime "$BASE" &&
    sbatch --parsable --job-name="eval_${SIZE}_base_wine" evaluate.sbatch wine "$BASE" &&
    sbatch --parsable --job-name="eval_${SIZE}_base_xsum" evaluate.sbatch xsum "$BASE" -T prompt_variant=long &&
    # The untrained model writes up to the 512-token cap, so it gets longer.
    sbatch --parsable --job-name="eval_${SIZE}_base_mt_domain_de_en" --time=03:00:00 \
      evaluate.sbatch mt_domain_de_en "$BASE")
}

all() {
  step "base" 4 base_evals || return 1
  for s in "${SEEDS[@]}"; do
    for task in crime wine mt_domain_de_en; do
      step "$task $s" 72 bash train/submit_lora_seed.sh "$task" "$s" --base "$BASE" --submit || return 1
    done
    step "xsum $s" 87 bash self-distill/submit_xsum_seed.sh "$s" --base "$BASE" --submit || return 1
  done
}
if all; then
  echo "$(date -u +%FT%TZ) all submitted"
else
  echo "$(date -u +%FT%TZ) queue full; next run in 10 minutes"
  sbatch --begin=now+600 submit_data_parallel_queue.sh "$BASE" "${SEEDS[@]}"
fi
