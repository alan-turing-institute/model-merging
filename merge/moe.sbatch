#!/bin/bash
#SBATCH --job-name=moe
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
# Usage, from merge/:
#   sbatch moe.sbatch <moe_config.yaml> <output_dir> [--seed <n>] [mergekit-moe args...]
#   e.g. sbatch moe.sbatch moe/mistral_task_moe_4_seed_1_config.yaml \
#            ../models/task_merge/seed_1/mistral_7b_task_moe_4 --seed 1
#
# Runs mergekit-moe (not mergekit-yaml) with --cuda --lazy-unpickle
# --allow-crimes: a sparse MoE (MixtralForCausalLM) whose experts are the
# config's source models' MLPs and whose every other weight is base_model's.
# gate_mode hidden runs base_model on the GPU over each expert's positive
# prompts to build the router. Anything after output_dir is passed on.
# mergekit-moe saves the config as <output_dir>/mergekit_moe_config.yml, and
# once the merge succeeds <output_dir>/provenance.json records each expert with
# its own record (see ../provenance.py). Relative paths, including those in the
# config, resolve against merge/. The log is saved as
# <output_dir>/<config name>-<jobid>.out.
#
# --seed (default 42) is mergekit-moe's --random-seed, recorded as random_seed.
# --router-prompts <moe/prompts_<model>.json>: after mergekit-moe, replace the
# router with one built from those example prompts (set_moe_router.py: gate_mode
# prompt_mean, described in <output_dir>/router.json).
# Run it online (HF_HUB_OFFLINE unset): mergekit lists a Hub base model's files.
set -euo pipefail
USAGE="usage: sbatch moe.sbatch <moe_config.yaml> <output_dir> [--seed <n>] [--router-prompts <json>] [mergekit-moe args...]"
CONFIG=${1:?$USAGE}
OUTPUT_DIR=${2:?$USAGE}
shift 2
SEED=42
ROUTER_PROMPTS=
ARGS=()
while (( $# )); do
  case $1 in
    --seed) SEED=${2:?$USAGE}; shift 2 ;;
    --router-prompts) ROUTER_PROMPTS=${2:?$USAGE}; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]}"
[[ $SEED =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer, not $SEED" >&2; exit 1; }

cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml && -f moe.sbatch ]] || { echo "submit from merge/, not $PWD" >&2; exit 1; }
[[ -f $CONFIG ]] || { echo "$CONFIG not found" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
RUN_NAME=$(basename "$CONFIG")
RUN_NAME=${RUN_NAME%.yaml}
RUN_NAME=${RUN_NAME%.yml}
exec > "$OUTPUT_DIR/${RUN_NAME}-${SLURM_JOB_ID}.out" 2>&1
trap '[[ -s slurm-$SLURM_JOB_ID.out ]] || rm -f "slurm-$SLURM_JOB_ID.out"' EXIT

# As in merge.sbatch: no CUDA compat lib, and the HF cache in the project area.
LD_LIBRARY_PATH=$(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run mergekit-moe "$CONFIG" "$OUTPUT_DIR" --cuda --lazy-unpickle --allow-crimes \
  --random-seed "$SEED" "$@"

if [[ -n $ROUTER_PROMPTS ]]; then
  uv run python set_moe_router.py "$OUTPUT_DIR" "$ROUTER_PROMPTS"
fi

# mergekit-moe copies base_model's config.json but writes no
# generation_config.json: see ../fix_eos.py.
uv run python ../fix_eos.py "$OUTPUT_DIR"

uv run python ../provenance.py model "$OUTPUT_DIR" --script moe.sbatch --random-seed "$SEED"
