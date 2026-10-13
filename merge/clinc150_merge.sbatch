#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
# Merge the two CLINC150 domain experts (screen arms expert_a, expert_b) with
# MergeKit, using the crime arm's settings so the two classification testbeds
# are merged the same way (merge_<method>_config.yaml): weight 0.5 each; TIES
# density 0.5; DARE-TIES density 0.3; base gemma-3-4b-it for the task-vector
# methods.
#
#   cd merge && sbatch clinc150_merge.sbatch <linear|ties|dare_ties|task_arithmetic> [seed]
#
# seed empty = the original experts; 2 or 3 = the seed reruns (expert_a_s2 ...).
# Merged models go to project storage: twelve 8 GB models would fill home.
set -euo pipefail
METHOD=${1:?usage: clinc150_merge.sbatch <method> [seed]}
SEED=${2:-}
SUF=${SEED:+_s$SEED}
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
export HF_HUB_OFFLINE=1
OUT=$PROJECTDIR/$USER/mm/models/clinc150-merged-$METHOD$SUF
[[ -f $OUT/config.json ]] && { echo "already merged: $OUT"; exit 0; }
rm -rf "$OUT"

A="google/gemma-3-4b-it+../models/gemma3-clinc150-expert_a$SUF-lora"
B="google/gemma-3-4b-it+../models/gemma3-clinc150-expert_b$SUF-lora"
case $METHOD in
  linear)          BASE="";                                   PARAMS="weight: 0.5" ;;
  ties)            BASE='base_model: "google/gemma-3-4b-it"'; PARAMS="weight: 0.5, density: 0.5" ;;
  dare_ties)       BASE='base_model: "google/gemma-3-4b-it"'; PARAMS="weight: 0.5, density: 0.3" ;;
  task_arithmetic) BASE='base_model: "google/gemma-3-4b-it"'; PARAMS="weight: 0.5" ;;
  *) echo "unknown method $METHOD" >&2; exit 1 ;;
esac
CFG=$(mktemp --suffix=.yaml -p .)
cat > "$CFG" <<YAML
merge_method: $METHOD
$BASE
models:
  - model: "$A"
    parameters: {$PARAMS}
  - model: "$B"
    parameters: {$PARAMS}
tokenizer_source: union
dtype: bfloat16
YAML
echo "=== $METHOD seed '${SEED:-1}' -> $OUT ==="; cat "$CFG"
uv run mergekit-yaml "$CFG" "$OUT" --cuda --lazy-unpickle --lora-merge-cache "${TMPDIR:-/tmp}/lora-cache-${SLURM_JOB_ID:-0}"
rm -f "$CFG"
