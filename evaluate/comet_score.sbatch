#!/bin/bash
#SBATCH --job-name=comet-score
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Usage, from evaluate/:
#   sbatch comet_score.sbatch <log.eval> [inspect score args...]
#   e.g. sbatch comet_score.sbatch logs/2026-10-16T10-00-00-00-00_translation_abc.eval
#        sbatch comet_score.sbatch logs/<log>.eval -S kiwi=false
#
# Appends COMET-22 and CometKiwi scores to a translation_task.py log, in place.
# Runs in the comet/ environment: unbabel-comet pins transformers<5 and
# numpy<2, so it is kept out of the environment that generates.
#
# Compute nodes have no outbound network, so the models must already be in
# $HF_HOME. Loading each once on a login node caches the checkpoints and their
# encoders' tokenizers (CometKiwi is gated: accept its licence on Hugging Face
# and `hf auth login` first, or pass -S kiwi=false):
#   HF_HOME=$PROJECTDIR/$USER/hf_cache uv run --project comet python -c \
#     "from comet import download_model as d, load_from_checkpoint as l; \
#      [l(d(m)) for m in ['Unbabel/wmt22-comet-da', 'Unbabel/wmt22-cometkiwi-da']]"
set -euo pipefail
USAGE="usage: sbatch comet_score.sbatch <log.eval> [inspect score args...]"
LOG=${1:?$USAGE}
shift

cd "$SLURM_SUBMIT_DIR"
[[ -f translation_task.py ]] || { echo "no translation_task.py in $PWD - submit from evaluate/" >&2; exit 1; }
[[ -f $LOG ]] || { echo "$LOG not found" >&2; exit 1; }

# See evaluate.sbatch: strip the CUDA forward-compat libcuda, which breaks the
# node driver with error 803.
LD_LIBRARY_PATH=$(tr ':' '
' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH

export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}
export HF_HUB_OFFLINE=1

uv run --project comet python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is a CUDA compat lib on LD_LIBRARY_PATH?'"

uv run --project comet inspect score "$LOG" --scorer translation_task.py@comet \
  --action append --overwrite --display plain "$@"
