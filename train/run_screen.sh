#!/bin/bash
# Submit a domain-split headroom screen: one training array over the task's arms,
# then an evaluation of each arm, plus the untrained base model.
#
#   ./run_screen.sh clinc150
#   ./run_screen.sh mt_domain
#   ./run_screen.sh clinc150 --eval-only
#
# Read the result with:  cd evaluate && uv run python split_screen.py logs --task <task>
set -euo pipefail
TASK=${1:?usage: run_screen.sh <clinc150|mt_domain> [--eval-only]}
EVAL_ONLY=0; [[ ${2:-} == --eval-only ]] && EVAL_ONLY=1
cd "$(dirname "$0")"
DATA_ROOT=../datasets/$TASK
[[ -f $DATA_ROOT/ARMS ]] || { echo "$DATA_ROOT/ARMS missing - run prepare_${TASK}_data.py on a login node" >&2; exit 1; }
mapfile -t ARMS < "$DATA_ROOT/ARMS"

# Per-task evaluation arguments. CLINC150's trained arms use the prompt they were
# trained on; its base model is scored on both variants, because without the
# label list an untrained model cannot know the label strings at all, and that
# zero is informative but is not its zero-shot ability.
case $TASK in
  clinc150)  TRAINED=(-T prompt_variant=short); BASES=("short" "labels"); EVAL_TIME=01:00:00 ;;
  mt_domain) TRAINED=();                        BASES=("");               EVAL_TIME=02:00:00 ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac

TRAIN_ID=
if (( ! EVAL_ONLY )); then
  TRAIN_ID=$(sbatch --parsable --job-name="$TASK" --array=0-$(( ${#ARMS[@]} - 1 )) train_screen.sbatch "$TASK")
  echo "training array: $TRAIN_ID (arms: ${ARMS[*]})"
fi

submit_eval() {
  local name=$1 dep=$2; shift 2
  local args=(--parsable --job-name="eval-$TASK-$name" --time="$EVAL_TIME")
  [[ -n $dep ]] && args+=(--dependency="$dep")
  ( cd ../evaluate && sbatch "${args[@]}" evaluate.sbatch "$TASK" google/gemma-3-4b-it "$@" )
}

for variant in "${BASES[@]}"; do
  extra=(); [[ -n $variant ]] && extra=(-T "prompt_variant=$variant")
  echo "  base${variant:+ ($variant)}: $(submit_eval "base${variant:+-$variant}" "" ${extra[@]+"${extra[@]}"})"
done
for i in "${!ARMS[@]}"; do
  dep=""; [[ -n $TRAIN_ID ]] && dep="afterok:${TRAIN_ID}_${i}"
  echo "  ${ARMS[$i]}: $(submit_eval "${ARMS[$i]}" "$dep" --adapter "../models/gemma3-$TASK-${ARMS[$i]}-lora" ${TRAINED[@]+"${TRAINED[@]}"})"
done
