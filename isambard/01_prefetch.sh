#!/bin/bash
# Pull every weight and dataset into the cache. LOGIN node only - compute nodes
# have no outbound network, and a job that discovers this mid-run fails after
# it has already queued and started.
set -euo pipefail

if [ "$(uname -s)" != "Linux" ] || ! command -v sinfo >/dev/null 2>&1; then
  echo "This runs on an Isambard LOGIN NODE, not here ($(hostname), $(uname -m))." >&2
  exit 1
fi

cd "$(dirname "$0")/.."
source isambard/config.sh
export HF_HUB_OFFLINE=0

uv run --project evaluate python - <<'PY'
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download("google/gemma-3-4b-it")   # needs `hf auth login` first; it is gated
load_dataset("EdinburghNLP/xsum")
print("cached")
PY

uv run --project evaluate python prepare_xsum_data.py
echo "Datasets built under $DATASETS_DIR"
