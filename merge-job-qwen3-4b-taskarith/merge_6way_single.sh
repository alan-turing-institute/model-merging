#!/bin/bash
set -euo pipefail
# One method per job. The combined version (merge_6way.sh) ran out of local disk
# on the cluster node: two rw_mount outputs plus six mounted checkpoints, with
# the first 8GB merge still on disk while the second was written. Splitting
# halves the peak.
METHOD="$1"; OUT="$2"; BASE="$3"; shift 3
EXPERTS=("$@")
W=$(python3 -c "print('%.6f' % (1.0/${#EXPERTS[@]}))")
echo "=== $METHOD over ${#EXPERTS[@]} experts at weight $W each ==="
df -h | head -12; echo "--- output filesystem ---"; df -h "$(dirname "$OUT")" 2>/dev/null || true

CFG=/tmp/${METHOD}-6way.yml
{
  echo "models:"
  for e in "${EXPERTS[@]}"; do
    echo "  - model: $e"
    echo "    parameters:"
    [ "$METHOD" = "ties" ] && echo "      density: 0.5"
    echo "      weight: $W"
  done
  echo "merge_method: $METHOD"
  if [ "$METHOD" = "ties" ]; then
    echo "base_model: $BASE"
    echo "parameters:"
    echo "  normalize: true"
    echo "  int8_mask: true"
  fi
  echo "dtype: bfloat16"
} > "$CFG"
cat "$CFG"

mergekit-yaml "$CFG" "$OUT" --cuda --lazy-unpickle --allow-crimes
cp "$CFG" "$OUT/mergekit_config.yml" 2>/dev/null || true
df -h | head -12; echo "--- output filesystem ---"; df -h "$(dirname "$OUT")" 2>/dev/null || true
echo "MERGE_COMPLETE"
