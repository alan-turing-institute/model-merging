#!/bin/bash
# One-time environment build on a LOGIN node (it needs the network).
#
# The hard part of this port is not Slurm, it is the architecture. Isambard-AI
# is GH200: aarch64 CPUs with Hopper GPUs. Several of axolotl's dependencies
# ship x86-only wheels and will either fail to install or fall back to a source
# build that needs nvcc and an hour of compute:
#
#   flash-attn      no aarch64 wheel; source build, ~45-90 min
#   bitsandbytes    aarch64 support exists from 0.43, but is the usual failure
#   xformers        often resolves to a source build
#
# QLoRA (4-bit) needs bitsandbytes. If it will not build, the fallback is plain
# LoRA in bf16 - a GH200 has 96 GB of HBM against the A100's 40, so a 4B model
# in bf16 with LoRA fits comfortably and 4-bit quantisation is not needed for
# capacity here. That changes the numbers slightly, so it must be recorded as a
# deviation rather than swapped in silently.
set -euo pipefail

# Refuse to run anywhere but a cluster login node. This has been run on a laptop
# twice: it succeeds, prints arch arm64 and "cuda available False", and none of
# that says anything about the GH200s. macOS reports arm64 where Linux reports
# aarch64, which is the tell.
if [ "$(uname -s)" != "Linux" ] || ! command -v sinfo >/dev/null 2>&1; then
  echo "This runs on an Isambard LOGIN NODE, not here." >&2
  echo "  host: $(hostname)   kernel: $(uname -s)   arch: $(uname -m)" >&2
  echo "  ssh in first, clone the repo there, then run this from the clone." >&2
  exit 1
fi

cd "$(dirname "$0")/.."
source isambard/config.sh

module load cuda 2>/dev/null || echo "no cuda module - check 'module avail cuda'"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

export HF_HUB_OFFLINE=0
for project in train merge evaluate; do
  echo "=== $project ==="
  uv sync --project "$project" || {
    echo "FAILED: $project. If it is flash-attn or bitsandbytes, see the note above." >&2
    exit 1
  }
done

echo
echo "=== verifying the GPU stack ==="
uv run --project train python - <<'PY'
import torch, platform
print("arch          ", platform.machine())
print("torch         ", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device        ", torch.cuda.get_device_name(0))
    print("capability    ", torch.cuda.get_device_capability(0))
for mod in ("bitsandbytes", "flash_attn"):
    try:
        __import__(mod); print(f"{mod:<14} ok")
    except Exception as exc:
        print(f"{mod:<14} MISSING ({type(exc).__name__})")
PY
