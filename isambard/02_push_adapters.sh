#!/bin/bash
# Copy the trained XSum adapters from THIS MACHINE to Isambard.
#
# Runs on the laptop, not the cluster - the inverse of every other script here.
#
# Why copy rather than retrain: the three XSum adapters already exist locally and
# are the exact artefacts every number in KD.md was measured against. Retraining
# them on Isambard would produce different weights (different seed, different
# hardware) and quietly break comparability with the halves arm on record. The
# KD re-run has to isolate the alignment fix; it cannot also change the experts.
#
# Only the adapter payload moves. optimizer.pt, scheduler.pt, rng_state.pth and
# the nested checkpoint directory are training state - roughly 290 MB per
# adapter of it - and nothing downstream reads them.
set -euo pipefail

: "${ISAMBARD_HOST:?set ISAMBARD_HOST, e.g. ISAMBARD_HOST=drmario.u6ui@login.isambard.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-model-merging/models}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/model-merging/models}"

ADAPTERS=(
  gemma3-xsum-1-of-2-lora
  gemma3-xsum-2-of-2-lora
  gemma3-xsum-full-lora
)

for adapter in "${ADAPTERS[@]}"; do
  [ -d "$LOCAL_ROOT/$adapter" ] || { echo "missing: $LOCAL_ROOT/$adapter" >&2; exit 1; }
  echo "=== $adapter ==="
  rsync -avh --progress \
    --include='adapter_config.json' \
    --include='adapter_model.safetensors' \
    --include='chat_template.jinja' \
    --include='tokenizer.json' \
    --include='tokenizer_config.json' \
    --include='tokens_state.json' \
    --include='README.md' \
    --exclude='*' \
    "$LOCAL_ROOT/$adapter/" "$ISAMBARD_HOST:$REMOTE_ROOT/$adapter/"
done

echo
echo "Done. ~77 MB each, ~230 MB total."
echo "On the cluster, confirm with:  ls -la $REMOTE_ROOT/*/adapter_model.safetensors"
