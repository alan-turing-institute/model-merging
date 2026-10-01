#!/bin/bash
# Submit the whole BillSum data-scaling curve: four training arms, then an
# evaluation of each, plus the untrained base model.
#
# Usage, from train/:
#   ./run_billsum_curve.sh               # submit, with dependencies
#   ./run_billsum_curve.sh --eval-only   # arms already trained, just score them
#
# THE BASE MODEL IS THE CURVE'S n=0 POINT, not a courtesy baseline. A curve drawn
# only through the four trained arms can look flat because the task is saturated,
# or flat because fine-tuning does nothing on it at all, and those have opposite
# implications for whether to run merging experiments here. The gap from base to
# the eighth arm is what tells them apart.
#
# aftercorr, not afterok: it pairs array index to array index, so the evaluation
# of the eighth arm starts when the eighth arm finishes rather than waiting for
# the full arm. On a busy queue that is most of the wall clock.
set -euo pipefail

EVAL_ONLY=0
[[ ${1:-} == --eval-only ]] && EVAL_ONLY=1

PROMPT_VARIANT=short            # must match prepare_billsum_data.py's default
LIMIT=500                       # test rows; the split holds 1000
EVAL_TIME=03:00:00              # BillSum generations are ~8x XSum's

cd "$(dirname "$0")"
EVAL_DIR=../evaluate
ARMS=(full half quarter eighth)

for arm in "${ARMS[@]}"; do
  [[ -d "../datasets/billsum/$arm/train" ]] || {
    echo "../datasets/billsum/$arm not found. On a LOGIN node (compute nodes have" >&2
    echo "no network), run:  cd train && uv run python prepare_billsum_data.py ../datasets/billsum" >&2
    exit 1
  }
done

TRAIN_ID=
if (( ! EVAL_ONLY )); then
  TRAIN_ID=$(sbatch --parsable train_billsum.sbatch)
  echo "training array: $TRAIN_ID (arms: ${ARMS[*]})"
fi

# Submitted from evaluate/, which is where evaluate.sbatch expects to run: the
# task files import semantic_metrics from there and relative model paths resolve
# against it.
submit_eval() {
  local name=$1 dep=$2; shift 2
  local args=(--parsable --job-name="eval-billsum-$name" --time="$EVAL_TIME")
  [[ -n $dep ]] && args+=(--dependency="$dep")
  ( cd "$EVAL_DIR" && sbatch "${args[@]}" evaluate.sbatch "$@" )
}

# n=0. No dependency: it can run immediately, and does not need the GPU the arms
# are queued for.
BASE_ID=$(submit_eval base "" billsum google/gemma-3-4b-it \
  -T prompt_variant=$PROMPT_VARIANT --limit $LIMIT)
echo "  base (n=0): $BASE_ID"

for i in "${!ARMS[@]}"; do
  arm=${ARMS[$i]}
  dep=""
  [[ -n $TRAIN_ID ]] && dep="aftercorr:${TRAIN_ID}_${i}"
  id=$(submit_eval "$arm" "$dep" billsum google/gemma-3-4b-it \
    --adapter "../models/gemma3-billsum-$arm-lora" \
    -T prompt_variant=$PROMPT_VARIANT --limit $LIMIT)
  echo "  $arm: $id${dep:+  (after $dep)}"
done

cat <<'EOF'

Logs land in evaluate/logs/ as .eval files. Read the curve with:
  cd evaluate && uv run inspect view
or collect the headline numbers with:
  cd evaluate && uv run python billsum_curve.py logs

Once the curve is in, the question it answers: is rougeLsum still rising between
half and full? If yes, BillSum has headroom and the merging arms are worth
running on it. If it has flattened, stop here - and that is a cheap answer, not a
failed experiment.
EOF
