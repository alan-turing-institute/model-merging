#!/bin/bash
#SBATCH --job-name=moe_routing
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=../evaluate/slurm/%x-%j.out
# Measures a task MoE's routing (moe_router_probe.py) on its evaluated inputs.
# Usage, from merge/:
#   sbatch moe_router_probe.sbatch <mistral|llama> <seed> [hidden|random|prompt_mean] [--limit N]
# Reads ../models/task_merge/seed_<S>/<stem>_task_moe_4[_random] and its result
# JSONs, ../results/task_merging/<model>/seed_<S>/moe[_random]/, and writes
# routing.json there.
set -euo pipefail
USAGE="usage: sbatch moe_router_probe.sbatch <mistral|llama> <seed> [hidden|random|prompt_mean] [--limit N]"
MODEL=${1:?$USAGE}; SEED=${2:?$USAGE}; GATE=${3:-hidden}; shift $(( $# < 3 ? $# : 3 ))
case $MODEL in mistral) STEM=mistral_7b ;; llama) STEM=llama3_8b ;; *) echo "$USAGE" >&2; exit 1 ;; esac
TAG=; [[ $GATE == hidden ]] || TAG=_$GATE
cd "$SLURM_SUBMIT_DIR"
[[ -f moe_router_probe.py ]] || { echo "submit from merge/, not $PWD" >&2; exit 1; }
MOE=../models/task_merge/seed_$SEED/${STEM}_task_moe_4$TAG
RESULTS=../results/task_merging/$MODEL/seed_$SEED/moe$TAG
[[ -f $MOE/mergekit_moe_config.yml ]] || { echo "no MoE at $MOE" >&2; exit 1; }
[[ -f $RESULTS/xsum.json ]] || { echo "no result JSONs in $RESULTS: run evaluate/export_results.py" >&2; exit 1; }

# As evaluate.sbatch: no CUDA compat lib, the HF cache in the project area, offline.
LD_LIBRARY_PATH=$(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH HF_HUB_OFFLINE=1 UV_NO_SYNC=1
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run --project ../evaluate python moe_router_probe.py "$MOE" "$RESULTS" --output "$RESULTS/routing.json" "$@"
