#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
# Merge N LoRA experts (each applied to gemma-3-4b-it) into one full model.
#   sbatch lora_merge.sbatch <out_name> <linear|ties|dare_ties|task_arithmetic> <adapter_dir> <adapter_dir> [...]
# An expert may instead be a full model directory (no adapter_config.json), as
# premerge_peft.py writes for Gemma 3 1B; set MERGE_BASE to the matching base
# directory for the task-vector methods.
# Writes $PROJECTDIR/$USER/mm/models/<out_name>. Every expert gets weight 1/N;
# TIES density 0.5, DARE-TIES density 0.3, as in clinc150_merge.sbatch and the
# crime configs. Task arithmetic at weight 1/N equals the linear average.
set -euo pipefail
NAME=${1:?usage: lora_merge.sbatch <out_name> <method> <adapter> <adapter> [...]}
METHOD=${2:?method}
shift 2
(( $# >= 2 )) || { echo "need at least two adapters" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
export HF_HUB_OFFLINE=1
# A bare repo id as base_model makes MergeKit list the repo tree, which fails
# offline, so the task-vector methods take the cached snapshot directory.
BASE_DIR=${MERGE_BASE:-$(ls -d "$HF_HUB_CACHE"/models--google--gemma-3-4b-it/snapshots/*/ | head -1)}
OUT=$PROJECTDIR/$USER/mm/models/$NAME
[[ -f $OUT/config.json ]] && { echo "already merged: $OUT"; exit 0; }
rm -rf "$OUT"
W=$(python3 -c "print(round(1/$#, 6))")
case $METHOD in
  linear)          BASE="";                          PARAMS="weight: $W" ;;
  ties)            BASE="base_model: \"$BASE_DIR\""; PARAMS="weight: $W, density: 0.5" ;;
  dare_ties)       BASE="base_model: \"$BASE_DIR\""; PARAMS="weight: $W, density: 0.3" ;;
  task_arithmetic) BASE="base_model: \"$BASE_DIR\""; PARAMS="weight: $W" ;;
  *) echo "unknown method $METHOD" >&2; exit 1 ;;
esac
CFG=$(mktemp --suffix=.yaml -p .)
{ echo "merge_method: $METHOD"; echo "$BASE"; echo "models:"
  for a in "$@"; do
    if [[ -f $a/adapter_config.json ]]; then m="google/gemma-3-4b-it+$a"; else m=$a; fi
    echo "  - model: \"$m\""; echo "    parameters: {$PARAMS}"; done
  echo "tokenizer_source: union"; echo "dtype: bfloat16"; } > "$CFG"
echo "=== $NAME ($METHOD, $# experts) ==="; cat "$CFG"
uv run mergekit-yaml "$CFG" "$OUT" --cuda --lazy-unpickle --lora-merge-cache "${TMPDIR:-/tmp}/lora-cache-${SLURM_JOB_ID:-0}"
rm -f "$CFG"
