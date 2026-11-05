#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
# Usage, from the repo root:
#   sbatch convert_to_full_model.sbatch <base_model> <adapter_dir> <output_dir>
#   e.g. sbatch convert_to_full_model.sbatch mistralai/Mistral-7B-Instruct-v0.3 \
#            models/crime/seed_1/mistral_7b_crime_mlp models/crime/seed_1/mistral_7b_crime_mlp_full
#
# Runs Convert_to_full_model.py in train/'s environment (transformers + peft):
# the LoRA adapter merged into its base and saved as a full model, in the base's
# dtype (bf16), with a provenance.json (see provenance.py) that keeps the
# adapter's record. The job fails if the merged weights hold NaN/inf. The merge
# runs on the CPU; the GPU is only what the partition allocates with a node.
# mergekit-moe needs full models, so this is how a LoRA expert gets into an MoE.
# The log is saved as <output_dir>/convert-<jobid>.out.
set -euo pipefail
USAGE="usage: sbatch convert_to_full_model.sbatch <base_model> <adapter_dir> <output_dir>"
BASE_MODEL=${1:?$USAGE}
ADAPTER_DIR=${2:?$USAGE}
OUTPUT_DIR=${3:?$USAGE}

cd "$SLURM_SUBMIT_DIR"
[[ -f Convert_to_full_model.py ]] || { echo "submit from the repo root, not $PWD" >&2; exit 1; }
[[ -f $ADAPTER_DIR/adapter_config.json ]] || { echo "$ADAPTER_DIR is not a LoRA adapter dir" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/convert-${SLURM_JOB_ID}.out" 2>&1
trap '[[ -s slurm-$SLURM_JOB_ID.out ]] || rm -f "slurm-$SLURM_JOB_ID.out"' EXIT

# As in train/train.sbatch: no CUDA compat lib, and the HF cache in the project area.
LD_LIBRARY_PATH=$(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run --project train python Convert_to_full_model.py "$BASE_MODEL" "$ADAPTER_DIR" "$OUTPUT_DIR"
