#!/bin/bash
set -euo pipefail
RESULTS="$1"
MODEL_PATH="$2"

echo "=== Model directory ==="
ls -la "$MODEL_PATH" | head -20
echo "=== config.json ==="
cat "$MODEL_PATH/config.json"

mkdir -p "$RESULTS"
echo "=== Router probe ==="
python router_probe.py --model "$MODEL_PATH" --output "$RESULTS/router_probe.json"
echo "PROBE_COMPLETE"
