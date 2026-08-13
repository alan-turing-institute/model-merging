#!/bin/bash
set -euo pipefail
CONFIG="$1"
MERGED_OUT="$2"
export HF_TOKEN="${3:-}"

echo "=== Disk layout ==="
df -h

echo "=== Merging: $CONFIG -> $MERGED_OUT ==="
mergekit-yaml "$CONFIG" "$MERGED_OUT" --cuda --lazy-unpickle --allow-crimes

echo "=== Final disk layout ==="
df -h
echo "MERGE_COMPLETE"
