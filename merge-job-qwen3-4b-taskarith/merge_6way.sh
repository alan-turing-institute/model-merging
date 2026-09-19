#!/bin/bash
set -euo pipefail
# Re-merges the Qwen3-4B pool WITHOUT the Chinese error-correction model, to
# separate two explanations for the n=7 collapse of ties (0.5147) and dare-ties
# (0.4264) against linear (0.7061) and task-arithmetic (0.7047) on identical
# 1/7 weights: does the sparsification break down at seven models, or is it
# fragile to one bad member?
#
# ChineseErrorCorrector4-4B scores 0.4409 on these English benchmarks because
# they do not test what it was trained for. Trim-and-sign-resolve gives every
# model a vote on which coordinates survive, so an outlier can plausibly
# dominate it in a way that plain averaging tolerates.
#
# Both methods are merged here, not just ties. Removing a weak member raises the
# benchmark mean for ANY method, so ties-6way improving proves nothing on its
# own - the attribution is in the ties-vs-linear GAP, which is 0.191 at n=7.
TIES_OUT="$1"; LINEAR_OUT="$2"; shift 2
BASE="$1"; shift
EXPERTS=("$@")

W=$(python3 -c "print('%.6f' % (1.0/${#EXPERTS[@]}))")
echo "=== ${#EXPERTS[@]} experts at weight $W each (sums to 1, as the 7-way configs did) ==="

write_config() {  # $1=method $2=outfile
  local method="$1" out="$2"
  {
    echo "models:"
    for e in "${EXPERTS[@]}"; do
      echo "  - model: $e"
      echo "    parameters:"
      [ "$method" = "ties" ] && echo "      density: 0.5"
      echo "      weight: $W"
    done
    echo "merge_method: $method"
    if [ "$method" = "ties" ]; then
      echo "base_model: $BASE"
      echo "parameters:"
      echo "  normalize: true"
      echo "  int8_mask: true"
    fi
    echo "dtype: bfloat16"
  } > "$out"
  echo "--- $out"; cat "$out"
}

write_config ties   /tmp/ties-6way.yml
write_config linear /tmp/linear-6way.yml

echo "=== merging ties (6-way) ==="
mergekit-yaml /tmp/ties-6way.yml "$TIES_OUT" --cuda --lazy-unpickle --allow-crimes
cp /tmp/ties-6way.yml "$TIES_OUT/mergekit_config.yml" 2>/dev/null || true

echo "=== merging linear (6-way) ==="
mergekit-yaml /tmp/linear-6way.yml "$LINEAR_OUT" --cuda --lazy-unpickle --allow-crimes
cp /tmp/linear-6way.yml "$LINEAR_OUT/mergekit_config.yml" 2>/dev/null || true

df -h
echo "MERGE_COMPLETE"
