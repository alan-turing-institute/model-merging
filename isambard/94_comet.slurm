#!/bin/bash
#SBATCH --job-name=comet-bakeoff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=115000M
#SBATCH --time=01:30:00
#SBATCH --output=logs/comet-%j.out
#SBATCH --error=logs/comet-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# COMET on the bake-off perturbations, in an isolated environment.
#
# WHY ISOLATED. unbabel-comet pulls pytorch-lightning and pins transformers back
# to 4.x. Installing it into evaluate/ downgraded the library every result in
# this programme was measured under - a metric is not worth changing the
# apparatus for. It therefore gets its own uv project, its own venv on project
# storage, and its own job.
#
# WHY A DUMP. It scores the triples 92_metric_bakeoff emitted rather than
# rebuilding them, so the two environments agree on a file instead of on an RNG
# call order. Regenerate with:
#   uv run --project evaluate python experiments/metric_bakeoff.py ... \
#     --dump-conditions $RESULTS_DIR/bakeoff-conditions.json
#
# WHAT IT TESTS. COMET is trained on human judgements of machine translation,
# where source, hypothesis and reference are length-matched and meaning
# equivalent. Summarisation gives it a 400-word article and a 20-word summary
# that is deliberately not equivalent to its source. Whether it still
# discriminates is the question, not an assumption.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no nvidia-smi)"

# As in evaluate/evaluate.sbatch: the nodes' driver (580.173) now supports the
# cu130 wheels natively, and the older CUDA 13.0 forward-compat libcuda (580.65)
# breaks it with error 803. sbatch copies the submitting shell's
# LD_LIBRARY_PATH, so strip any compat dir from it.
LD_LIBRARY_PATH=$(tr ':' '
' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH

# The venv lives wherever the caller keeps it. It used to be hardcoded to one
# user's project directory, which meant the job only ran for that user.
export UV_PROJECT_ENVIRONMENT="${METRICS_VENV:-$REPO_ROOT/metrics/.venv}"
[ -f "$RESULTS_DIR/bakeoff-conditions.json" ] || { echo "no condition dump" >&2; exit 1; }
[ -d "$HF_HUB_CACHE/models--Unbabel--wmt22-comet-da" ] || { echo "comet not prefetched" >&2; exit 1; }

uv run --project metrics python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is a CUDA compat lib on LD_LIBRARY_PATH?'"
uv run --project metrics python experiments/score_comet.py \
  --conditions "$RESULTS_DIR/bakeoff-conditions.json" \
  --output     "$RESULTS_DIR/comet-bakeoff.json"
