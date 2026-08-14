#!/bin/bash
set -euo pipefail
CONFIG="$1"
MERGED_OUT="$2"

# Redirect the HF cache to whichever mount has more room. On the A100 SKU
# used previously (NC24ads_A100_v4) the default container disk was far
# smaller than the spec sheet implied; /mnt sometimes has more headroom.
if mkdir -p /mnt/hf_cache 2>/dev/null && [ -w /mnt/hf_cache ]; then
  export HF_HOME=/mnt/hf_cache
fi

echo "=== Disk layout ==="
df -h
echo "HF_HOME=${HF_HOME:-default}"

echo "=== Merging: $CONFIG -> $MERGED_OUT ==="
mergekit-yaml "$CONFIG" "$MERGED_OUT" --cuda --lazy-unpickle --allow-crimes

echo "=== Final disk layout ==="
df -h
echo "MERGE_COMPLETE"
