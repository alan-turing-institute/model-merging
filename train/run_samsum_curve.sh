#!/bin/bash
# Submit the whole SAMSum data-scaling curve: six training arms, then an
# evaluation of each, plus the untrained base model.
#
# Usage, from train/:
#   ./run_samsum_curve.sh               # submit, with dependencies
#   ./run_samsum_curve.sh --eval-only   # arms already trained, just score them
#
# THE BASE MODEL IS THE CURVE'S n=0 POINT, not a courtesy baseline. A curve drawn
# only through the trained arms can look flat because the task is saturated, or
# flat because fine-tuning does nothing on it at all, and those have opposite
# implications. The gap from base to the 250-row arm is what tells them apart -
# and on SAMSum it is the measurement the whole screen turns on, because the
# expected shape here is a large base->250 jump followed by a flat curve, which
# would mean the task teaches FORMAT rather than capability.
#
# Each evaluation depends on ITS OWN array task (afterok:<jobid>_<index>), not on
# the array as a whole, so the smallest arm is scored as soon as it finishes
# rather than waiting for the largest.
set -euo pipefail

EVAL_ONLY=0
[[ ${1:-} == --eval-only ]] && EVAL_ONLY=1

PROMPT_VARIANT=short            # must match prepare_samsum_data.py's default
LIMIT=500                       # test rows; the split holds 819, we prepared 500
EVAL_TIME=01:00:00              # SAMSum generations are ~1/5 of BillSum's

cd "$(dirname "$0")"
EVAL_DIR=../evaluate
# Must match the ARMS array in train_samsum.sbatch, index for index: the
# evaluation dependencies are built from these positions.
ARMS=(n8000 n4000 n2000 n1000 n500 n250)

for arm in "${ARMS[@]}"; do
  [[ -d "../datasets/samsum/$arm/train" ]] || {
    echo "../datasets/samsum/$arm not found. On a LOGIN node (compute nodes have" >&2
    echo "no network), run:  cd train && uv run python prepare_samsum_data.py ../datasets/samsum" >&2
    exit 1
  }
done

TRAIN_ID=
if (( ! EVAL_ONLY )); then
  TRAIN_ID=$(sbatch --parsable train_samsum.sbatch)
  echo "training array: $TRAIN_ID (arms: ${ARMS[*]})"
fi

# Submitted from evaluate/, which is where evaluate.sbatch expects to run: the
# task files import semantic_metrics from there and relative model paths resolve
# against it.
submit_eval() {
  local name=$1 dep=$2; shift 2
  local args=(--parsable --job-name="eval-samsum-$name" --time="$EVAL_TIME")
  [[ -n $dep ]] && args+=(--dependency="$dep")
  ( cd "$EVAL_DIR" && sbatch "${args[@]}" evaluate.sbatch "$@" )
}

# n=0. No dependency: it can run immediately.
BASE_ID=$(submit_eval base "" samsum google/gemma-3-4b-it \
  -T prompt_variant=$PROMPT_VARIANT --limit $LIMIT)
echo "  base (n=0): $BASE_ID"

for i in "${!ARMS[@]}"; do
  arm=${ARMS[$i]}
  dep=""
  [[ -n $TRAIN_ID ]] && dep="afterok:${TRAIN_ID}_${i}"
  id=$(submit_eval "$arm" "$dep" samsum google/gemma-3-4b-it \
    --adapter "../models/gemma3-samsum-$arm-lora" \
    -T prompt_variant=$PROMPT_VARIANT --limit $LIMIT)
  echo "  $arm: $id${dep:+  (after $dep)}"
done

cat <<'EOF'

Logs land in evaluate/logs/ as .eval files. Read the curve with:
  cd evaluate && uv run python scaling_curve.py logs --task samsum \
      --json ../results/samsum-scaling-curve.json

Two questions it answers, and they are not the same question:

  1. SATURATION - is rouge1 still rising at the top of the curve? If it has
     flattened, no amount of data helps and SAMSum joins XSum on the shelf.

  2. MERGING HEADROOM - the "denominator of R" table. For a two-way split, the
     denominator is the step from n/2 to n, and R can only be estimated to about
     +/-0.1 if that step is ~10x the seed spread (0.005, so ~0.05). A curve can
     be flat at the top and STILL have a usable denominator lower down; that
     operating point is the one to run the merging arms at.

If neither question comes out well, stop here - that is a cheap answer after
about an hour of GPU time, not a failed experiment.
EOF
