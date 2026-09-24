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

# cuda/12.6 against cuda/11.8; torch wheels for aarch64 are built for cu12.
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null \
  || echo "no cuda module loaded - check 'module avail cuda'"
# uv installs to ~/.local/bin, which is not on PATH in a fresh login shell -
# so a first-run install succeeds and then "uv: command not found" on the very
# next line. Put it on PATH here rather than telling the user to re-run.
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "uv still not on PATH - source ~/.local/bin/env" >&2; exit 1; }

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
# All three projects, not just train: each resolves torch independently, and a
# cu130 build in any one of them is a silent CPU fallback in whichever step uses
# it - the merge and the evaluation are as GPU-bound as the training.
for project in train evaluate merge; do
  echo "--- $project ---"
  uv run --project "$project" python - <<'PYCHECK'
import platform
import torch

print("arch          ", platform.machine())
print("torch         ", torch.__version__)
# A cu130 build cannot initialise against this cluster's 12.7 driver. That shows
# up as cuda available False with a "driver is too old" warning - which on a
# login node is indistinguishable from simply having no GPU, so check the build.
if "cu130" in torch.__version__:
    print("BUILD          WRONG - cu130 will not run here; expected +cu128")
# Login nodes have no GPU, so False is expected here and says nothing about the
# compute nodes. Only the build tag above is conclusive from a login node.
print("cuda available", torch.cuda.is_available(),
      "" if torch.cuda.is_available() else "(expected on a login node)")
if torch.cuda.is_available():
    print("device        ", torch.cuda.get_device_name(0))
PYCHECK
done

echo "--- train extras ---"
# These two are the whole question on aarch64. bitsandbytes gates load_in_4bit;
# flash_attn only costs speed, since axolotl falls back to eager attention.
uv run --project train python - <<'PYEXTRA'
for mod in ("bitsandbytes", "flash_attn"):
    try:
        __import__(mod)
        print(f"{mod:<14} ok")
    except Exception as exc:
        print(f"{mod:<14} MISSING ({type(exc).__name__})")
PYEXTRA
