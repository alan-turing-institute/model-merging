#!/bin/bash
# Submit the CNN/DailyMail data-scaling curve for one base model: one training
# array over the arms in ../datasets/cnndm_curve/ARMS, then an evaluation of
# each arm as soon as its own array task finishes.
#
# Usage, from train/, after prepare_cnndm_data.py --ladder on a login node:
#   ./run_cnndm_curve.sh                          # gemma-3-4b-it
#   ./run_cnndm_curve.sh google/gemma-3-1b-it
#   ./run_cnndm_curve.sh google/gemma-3-1b-it --eval-only
#
# The n=0 point is the untrained model's score from the CNN/DailyMail screen
# (eval-cnndm-base-4b / -1b): same test rows, prompt and generation cap, so it
# is not re-run. Read the curve with:
#   cd ../evaluate && uv run --no-sync python scaling_curve.py logs --task cnndm \
#       --model gemma-3-4b-it --json ../results/cnndm-scaling-curve-4b.json
set -euo pipefail
BASE_MODEL=${1:-google/gemma-3-4b-it}
EVAL_ONLY=0; [[ ${2:-} == --eval-only ]] && EVAL_ONLY=1
cd "$(dirname "$0")"
DATA_ROOT=../datasets/cnndm_curve
[[ -f $DATA_ROOT/ARMS ]] || { echo "$DATA_ROOT/ARMS missing - run prepare_cnndm_data.py --ladder on a login node" >&2; exit 1; }
mapfile -t ARMS < "$DATA_ROOT/ARMS"

# The 4B is train_screen.sbatch's default model and gets no size suffix, so
# pass it through only when it is not the default.
SIZE=$(grep -oiE '[0-9]+b' <<<"${BASE_MODEL##*/}" | head -1 | tr 'B' 'b')
MODEL_ARG=""; SUFFIX=""
[[ $BASE_MODEL != google/gemma-3-4b-it ]] && { MODEL_ARG=$BASE_MODEL; SUFFIX=_$SIZE; }

TRAIN_ID=
if (( ! EVAL_ONLY )); then
  TRAIN_ID=$(sbatch --parsable --job-name="cnndm-curve-$SIZE" --array=0-$(( ${#ARMS[@]} - 1 )) \
    train_screen.sbatch cnndm_curve "" $MODEL_ARG)
  echo "training array: $TRAIN_ID (arms: ${ARMS[*]})"
fi

for i in "${!ARMS[@]}"; do
  dep=""; [[ -n $TRAIN_ID ]] && dep="afterok:${TRAIN_ID}_${i}"
  id=$(cd ../evaluate && sbatch --parsable --job-name="eval-cnndm-${ARMS[$i]}-$SIZE" --time=02:00:00 \
    ${dep:+--dependency=$dep} evaluate.sbatch cnndm "$BASE_MODEL" \
    --adapter "../models/gemma3-cnndm_curve-${ARMS[$i]}$SUFFIX-lora")
  echo "  ${ARMS[$i]}: $id${dep:+  (after $dep)}"
done
