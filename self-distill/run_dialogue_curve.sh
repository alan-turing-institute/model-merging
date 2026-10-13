#!/bin/bash
# Submit one dialogue-task scaling curve for one base model: a training array
# over ../datasets/dlg_<task>_curve/ARMS (train/train_screen.sbatch), the
# untrained model's score, and each arm's score once its array task finishes.
#   ./run_dialogue_curve.sh reorder                         # gemma-3-4b-it
#   ./run_dialogue_curve.sh constrained google/gemma-3-1b-it
#   ./run_dialogue_curve.sh constrained_sft      # arms from dialogue_data.py build --name constrained_sft
# Read with:  python dialogue_curve.py ../results/dialogue_curves/<task> _4b
set -euo pipefail
TASK=${1:?usage: run_dialogue_curve.sh <reorder|constrained|constrained_sft> [base_model]}
BASE_MODEL=${2:-google/gemma-3-4b-it}
cd "$(dirname "$0")"
DATA=../datasets/dlg_${TASK}_curve
[[ -f $DATA/ARMS ]] || { echo "$DATA/ARMS missing - run dialogue_data.py build $TASK" >&2; exit 1; }
mapfile -t ARMS < "$DATA/ARMS"
SIZE=$(grep -oiE '[0-9]+b' <<<"${BASE_MODEL##*/}" | head -1 | tr 'B' 'b')
MODEL_ARG=""; SUFFIX=""
[[ $BASE_MODEL != google/gemma-3-4b-it ]] && { MODEL_ARG=$BASE_MODEL; SUFFIX=_$SIZE; }

TRAIN_ID=$(cd ../train && sbatch --parsable --job-name="dlg-$TASK-$SIZE" --array=0-$(( ${#ARMS[@]} - 1 )) \
  train_screen.sbatch "dlg_${TASK}_curve" "" $MODEL_ARG)
echo "training array: $TRAIN_ID (arms: ${ARMS[*]})"
echo "  base: $(sbatch --parsable --job-name="eval-dlg-$TASK-base-$SIZE" dialogue.sbatch eval "$BASE_MODEL" "$TASK" "base_$SIZE")"
for i in "${!ARMS[@]}"; do
  echo "  ${ARMS[$i]}: $(sbatch --parsable --job-name="eval-dlg-$TASK-${ARMS[$i]}-$SIZE" \
    --dependency="afterok:${TRAIN_ID}_${i}" dialogue.sbatch eval "$BASE_MODEL" "$TASK" "${ARMS[$i]}_$SIZE" \
    "../models/gemma3-dlg_${TASK}_curve-${ARMS[$i]}$SUFFIX-lora")"
done
