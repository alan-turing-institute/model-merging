#!/bin/bash
#SBATCH --job-name=mistral-base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/mistral-base-%j.out
#SBATCH --error=logs/mistral-base-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# UNTUNED Mistral-7B-Instruct-v0.3 on the same two tasks the MoE and its
# parents were measured on, with the identical protocol from 82_mistral_moe
# (crime: --limit 200 --max-new-tokens 16; xsum: --limit 200, default cap).
#
# Supplies the only missing term in
#     retained over base = (moe - parent_base) / (parent - parent_base)
# which Appendix A.11 currently reports as uncomputable.
#
# PREREQUISITE - the weights must be resident before this is submitted. They
# were in HF_HUB_CACHE when 82_mistral_moe ran and were later removed, which is
# how job 7113108 failed six seconds in: compute nodes have no outbound network,
# so a missing snapshot is only discovered after the job has queued and started.
# On the LOGIN node, before sbatch:
#
#   bash ~/prefetch_mistral.sh        # or:
#   source isambard/config.sh && HF_HUB_OFFLINE=0 \
#     uv run --project evaluate python -c \
#     "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-Instruct-v0.3')"
#
# Verify with: ls -d $HF_HUB_CACHE/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

BASE=$(ls -d "$HF_HUB_CACHE"/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/ | head -1)
BASE="${BASE%/}"
echo "=== $(hostname) | $(date -Is) ==="
echo "base: $BASE"

cd "$REPO_ROOT/evaluate"
uv run python evaluate.py --model "$BASE" \
  --dataset "$DATASETS_DIR/crime_dataset/test" --limit 200 --max-new-tokens 16 \
  --output "$RESULTS_DIR/mistral-base-crime.json" 2>&1 | grep -E "^Accuracy|^Unparsed" || true
uv run python evaluate_summarisation.py --model "$BASE" \
  --dataset "$DATASETS_DIR/xsum_dataset/test" --limit 200 \
  --output "$RESULTS_DIR/mistral-base-xsum.json" 2>&1 | grep -E "^rouge1|Length" || true

echo "=== done $(date -Is) ==="
