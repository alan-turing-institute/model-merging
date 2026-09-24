#!/bin/bash
#SBATCH --job-name=xsum-kd8-concat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/kd8-concat-%j.out
#SBATCH --error=logs/kd8-concat-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Combine the eight per-shard teacher datasets into the one the sweep trains
# on. Its own job rather than a step inside the sweep: three sweep tasks start
# at once and would race to write the same directory.
#
# CPU-only work holding a GPU, because one GPU is the minimum allocatable unit
# here. It is half an hour at the outside.
#
#   sbatch --dependency=afterok:<teachers-array-id> isambard/41_kd8_concat.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs

SPLITS="${SPLITS:-8}"
OUT="$DATASETS_DIR/xsum_kd${SPLITS}_train"

if [ -d "$OUT" ]; then echo "=== already at $OUT ==="; exit 0; fi

INPUTS=()
for i in $(seq 1 "$SPLITS"); do
  d="$DATASETS_DIR/xsum_kd${SPLITS}_shard${i}"
  # Refuse a pre-fix shard rather than concatenate it into the training set.
  [ -f "$d/.kd_alignment_v2" ] \
    || { echo "$d missing .kd_alignment_v2 - not written by the corrected producer" >&2; exit 1; }
  INPUTS+=("$d")
done

echo "=== concatenating ${#INPUTS[@]} shards -> $OUT | $(date -Is) ==="
uv run --project evaluate python concat_datasets.py "${INPUTS[@]}" --output "$OUT"
echo "=== done $(date -Is) ==="
