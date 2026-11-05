#!/bin/bash
# Evaluates each MLP-only task expert (the inputs to task merging) on the three
# tasks it was NOT trained on, so the expert rows of results/task-merging.md
# can be filled in: 3 models x 5 seeds x 4 experts x 3 tasks = 180 jobs.
# Each is evaluated exactly as its own-task eval was: the LoRA experts (crime,
# wine, CLINC150) as base + --adapter, the XSum expert as a full model; CLINC150
# with -T prompt_variant=labels and XSum with -T prompt_variant=long.
#
# Usage, from evaluate/:
#   bash submit_expert_cross_evals.sh [--models "gemma mistral llama"] [--seeds "1 2 3 4 5"]
#        [--only <file of job names>] [--submit]
# Without --submit it only prints what it would submit. --only submits just the
# jobs named in the file (one per line), e.g. to rerun failed ones.
# CLINC150 (4,500 queries) gets 3 h: off-task experts take 30-60+ min on it,
# and 20 of 60 hit evaluate.sbatch's 1 h default on 5 Nov.
set -euo pipefail
cd "$(dirname "$0")"
MODELS="gemma mistral llama"; SEEDS="1 2 3 4 5"; SUBMIT=false; ONLY=
while [[ $# -gt 0 ]]; do
  case $1 in
    --models) MODELS=$2; shift ;;
    --seeds) SEEDS=$2; shift ;;
    --only) ONLY=$(realpath "$2"); shift ;;
    --submit) SUBMIT=true ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
  shift
done
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
JOB_MAP=slurm/expert_cross_evals_jobs_$(date +%F).txt
declare -A BASE=([gemma]=google/gemma-3-4b-it [mistral]=mistralai/Mistral-7B-Instruct-v0.3
                 [llama]=meta-llama/Llama-3.1-8B-Instruct)
declare -A STEM=([gemma]=gemma3_4b [mistral]=mistral_7b [llama]=llama3_8b)
declare -A EXPERT=([crime]=crime_mlp [wine]=wine_mlp [clinc150]=clinc150_labels_mlp [xsum]=xsum_gold_long_mlp)
declare -A TIME=([crime]=01:00:00 [wine]=01:00:00 [clinc150]=03:00:00 [xsum]=01:00:00)
declare -A EXTRA=([crime]="" [wine]="" [clinc150]="-T prompt_variant=labels" [xsum]="-T prompt_variant=long")
TASKS="crime wine clinc150 xsum"

n=0
for model in $MODELS; do
  for seed in $SEEDS; do
    for expert in $TASKS; do
      dir=../models/$expert/seed_$seed/${STEM[$model]}_${EXPERT[$expert]}
      [[ -f $dir/provenance.json ]] || { echo "missing $dir/provenance.json" >&2; exit 1; }
      if [[ $expert == xsum ]]; then target=("$dir"); else target=("${BASE[$model]}" --adapter "$dir"); fi
      for task in $TASKS; do
        [[ $task == "$expert" ]] && continue
        name=eval_x_${expert}_on_${task}_${model}_s$seed
        [[ -n $ONLY ]] && ! grep -qx "$name" "$ONLY" && continue
        # shellcheck disable=SC2086  # EXTRA is "" or "-T prompt_variant=..."
        cmd=(sbatch --parsable --job-name="$name" --time="${TIME[$task]}" evaluate.sbatch "$task" "${target[@]}" ${EXTRA[$task]})
        if $SUBMIT; then
          job=$("${cmd[@]}")
          echo "$job $name" | tee -a "$JOB_MAP"
        else
          echo "  ${cmd[*]}"
        fi
        n=$((n + 1))
      done
    done
  done
done
if $SUBMIT; then echo "Submitted $n jobs; job map: $JOB_MAP"; else echo "Dry run: $n jobs. Rerun with --submit."; fi
