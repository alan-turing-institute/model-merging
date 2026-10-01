#!/bin/bash
#SBATCH --job-name=cuda-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=00:15:00
#SBATCH --output=logs/cuda-probe-%j.out
#SBATCH --error=logs/cuda-probe-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Re-test the CUDA build choice instead of inheriting it from a comment.
#
# train/pyproject.toml documents two routes and takes the second:
#   1. torch from PyPI (cu130) plus the team's CUDA 13 forward-compatibility
#      libraries on LD_LIBRARY_PATH
#   2. torch from the cu128 index, matching the 12.7 driver directly through
#      CUDA minor-version compatibility
#
# The argument for 2 was that 1 fails SILENTLY when the variable is missing:
# torch falls back to CPU and training still completes, only slower. That
# argument holds while the driver is CUDA 12.x. If Isambard has moved to a 13.x
# driver then cu130 is native and cu128 becomes the fragile choice. This
# measures the driver and tests both builds, with and without the compat libs.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
PROJ=/projects/u6ui/drmario.u6ui/mm
COMPAT=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat

echo "=== driver and GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "  nvidia-smi unavailable"

echo
echo "=== the build in use: cu128, no compat libs ==="
UV_NO_SYNC=1 uv run --project evaluate python - <<'PY' 2>&1 | grep -vE "Installed|Uninstalled" | tail -4
import torch
print("  torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("  cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    torch.randn(512, 512, device="cuda")
    print("  matmul ok on", torch.cuda.get_device_name(0))
PY

echo
echo "=== the rejected build: cu130, no compat libs ==="
cd "$PROJ/cuda130test"
UV_PROJECT_ENVIRONMENT="$PROJ/venvs/cuda130" UV_NO_SYNC=1 uv run python - <<'PY' 2>&1 | tail -3
import torch
print("  torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("  cuda.is_available():", torch.cuda.is_available())
PY

echo
echo "=== cu130 WITH the CUDA 13 compat libraries ==="
[ -d "$COMPAT" ] && echo "  using $COMPAT" || echo "  compat path MISSING"
LD_LIBRARY_PATH="$COMPAT:${LD_LIBRARY_PATH:-}" \
UV_PROJECT_ENVIRONMENT="$PROJ/venvs/cuda130" UV_NO_SYNC=1 uv run python - <<'PY' 2>&1 | tail -3
import torch
print("  cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    torch.randn(512, 512, device="cuda")
    print("  matmul ok on", torch.cuda.get_device_name(0))
PY
