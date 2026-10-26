#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
# Usage, from merge/:
#   sbatch merge.sbatch <config.yaml> <output_dir> [mergekit-yaml args...]
#   e.g. sbatch merge.sbatch xsum_merge_linear_config.yaml \
#            ../models/gemma3_4b_xsum_short_merge_linear
#        sbatch merge.sbatch crime_merge_ties_config.yaml ../models/gemma3_4b_crime_ties \
#            --lora-merge-cache ../models/lora_merge_cache
#
# Runs mergekit-yaml with --cuda --lazy-unpickle --allow-crimes; anything after
# output_dir is passed on too (e.g. --lora-merge-cache, needed when the config
# uses the <base>+<adapter> syntax; see README.md). mergekit saves the config as
# <output_dir>/mergekit_config.yml, and once the merge succeeds
# <output_dir>/provenance.json records each input model with its weight and its
# own record (see ../provenance.py). Relative paths, including those in the
# config, resolve against merge/. The log is saved as
# <output_dir>/<config name>-<jobid>.out.
set -euo pipefail
USAGE="usage: sbatch merge.sbatch <config.yaml> <output_dir> [mergekit-yaml args...]"
CONFIG=${1:?$USAGE}
OUTPUT_DIR=${2:?$USAGE}
shift 2

# Submit from merge/: uv needs its pyproject.toml to find the environment.
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml && -f merge.sbatch ]] || { echo "submit from merge/, not $PWD" >&2; exit 1; }
[[ -f $CONFIG ]] || { echo "$CONFIG not found" >&2; exit 1; }

# #SBATCH directives can't see script arguments, so send stdout/stderr to the
# output dir from here. Anything printed before this line (e.g. a usage error)
# lands in Slurm's default slurm-<jobid>.out in the submit dir.
mkdir -p "$OUTPUT_DIR"
RUN_NAME=$(basename "$CONFIG")
RUN_NAME=${RUN_NAME%.yaml}
RUN_NAME=${RUN_NAME%.yml}
exec > "$OUTPUT_DIR/${RUN_NAME}-${SLURM_JOB_ID}.out" 2>&1
# Nothing more goes to slurm-<jobid>.out, so on exit delete it if it is empty.
# Kept if it holds an early error or a message from Slurm (e.g. out of memory).
trap '[[ -s slurm-$SLURM_JOB_ID.out ]] || rm -f "slurm-$SLURM_JOB_ID.out"' EXIT

# The nodes' driver (580.173) now supports the cu130 wheels natively, and the
# older CUDA 13.0 forward-compat libcuda (580.65) breaks it with error 803. sbatch
# copies the submitting shell's LD_LIBRARY_PATH, so strip the compat dir from it.
LD_LIBRARY_PATH=$(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH

# Hugging Face downloads (base models) live in the project area, not the
# 100 GB home quota. See the main README.
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run mergekit-yaml "$CONFIG" "$OUTPUT_DIR" --cuda --lazy-unpickle --allow-crimes "$@"

# mergekit writes no generation_config.json, and a config.json copied from a
# trained model has lost Gemma's <end_of_turn> stop token: see ../fix_eos.py.
uv run python ../fix_eos.py "$OUTPUT_DIR"

uv run python ../provenance.py model "$OUTPUT_DIR" --script merge.sbatch
