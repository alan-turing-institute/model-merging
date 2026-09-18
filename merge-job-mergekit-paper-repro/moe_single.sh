#!/bin/bash
set -euo pipefail
CONFIG="$1"
MERGED_OUT="$2"
export HF_TOKEN="${3:-}"

echo "=== Disk layout ==="
df -h

# mergekit-moe, not mergekit-yaml: a different entry point with its own config
# schema (experts/gate_mode rather than models/merge_method). gate_mode: hidden
# runs a forward pass over the positive and negative prompts for every layer, so
# this needs a GPU with room for one 7B model plus activations - hence the A100
# target rather than the T4 the dense merges use.
echo "=== MoE merge: $CONFIG -> $MERGED_OUT ==="
mergekit-moe "$CONFIG" "$MERGED_OUT" --cuda --lazy-unpickle --allow-crimes

echo "=== Output config sanity ==="
# The vocab mismatch (32,000 vs 32,017) is what corrupted embed_tokens and
# lm_head in the LERP and DARE-TIES reproductions. Check what this one declares
# before spending an evaluation on it.
python - "$MERGED_OUT" <<'PY'
import json, sys, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
for k in ("model_type", "vocab_size", "num_local_experts",
          "num_experts_per_tok", "architectures"):
    print(f"  {k}: {cfg.get(k)}")
if cfg.get("vocab_size") != 32000:
    print("  WARNING: expected vocab_size 32000 from the Llama-2 base")
PY

echo "=== Final disk layout ==="
df -h
echo "MERGE_COMPLETE"
