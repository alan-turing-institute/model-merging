#!/bin/bash
#SBATCH --job-name=qwen3-moe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/qwen3-moe-%j.out
#SBATCH --error=logs/qwen3-moe-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Build a two-expert sparse MoE from the Qwen3 experts, and evaluate it against
# both parents on both tasks.
#
# WHY QWEN3. mergekit-moe emits Mixtral, DeepSeek, Qwen and Qwen3 architectures
# and has no Gemma path, so the Gemma-3 experts every other result uses cannot
# be routed. Nothing here compares with a Gemma-3 number.
#
# WHY MATERIALISE FIRST. mergekit-moe takes full models: there is no .merged()
# call or loader cache in mergekit/moe/, so base+adapter composition is not
# available on that path. Each expert is written out once, ~8 GB, to node-local
# scratch - the home quota is 101 GB and has stopped jobs twice this week.
#
# WHAT IT CAN AND CANNOT SHOW. Cross-task merging already retains 99.6% and
# 98.6% of the two abilities, so this configuration has almost no headroom. It
# is the reference case. The informative case is same-direction experts, where
# merging fails outright.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

WORK="$TMPDIR/qwen3-moe"
mkdir -p "$WORK"
XSUM_FULL="$WORK/qwen3-xsum-expert"
CRIME_FULL="$WORK/qwen3-crime-expert"
MOE_OUT="$WORK/qwen3-moe-2way"

echo "=== $(hostname) | $(date -Is) | scratch $WORK ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || echo "  (nvidia-smi unavailable on this node)"
df -h "$TMPDIR" | tail -1

# --- 1. materialise each expert as a full model ---------------------------
cd "$REPO_ROOT/merge"
for spec in "xsum_full:$XSUM_FULL" "crime_full:$CRIME_FULL"; do
  tag="${spec%%:*}"; out="${spec#*:}"
  if [ -f "$out/config.json" ]; then echo "=== $tag already materialised ==="; continue; fi
  rm -rf "$out"
  echo "=== materialising $tag ==="
  uv run mergekit-yaml "merge_qwen3_materialise_${tag}.yaml" "$out" \
    --cuda --lazy-unpickle --lora-merge-cache "$TMPDIR/lora-$SLURM_JOB_ID"
done

# --- 2. build the MoE ------------------------------------------------------
CFG="$WORK/moe.yml"
sed -e "s|MATERIALISED_XSUM|$XSUM_FULL|" -e "s|MATERIALISED_CRIME|$CRIME_FULL|" \
  "$REPO_ROOT/merge/moe_qwen3_2way.yml" > "$CFG"
if [ ! -f "$MOE_OUT/config.json" ]; then
  rm -rf "$MOE_OUT"
  echo "=== mergekit-moe ==="
  uv run mergekit-moe "$CFG" "$MOE_OUT" --cuda --lazy-unpickle --allow-crimes
fi
echo "=== MoE config ==="
python3 -c "
import json;c=json.load(open('$MOE_OUT/config.json'))
print({k:c.get(k) for k in ('model_type','num_local_experts','num_experts_per_tok','vocab_size')})"

# --- 3. evaluate the MoE and both parents on both tasks --------------------
cd "$REPO_ROOT/evaluate"
echo "=== summarisation ==="
for spec in "moe:$MOE_OUT" "xsum-expert:$XSUM_FULL" "crime-expert:$CRIME_FULL"; do
  n="${spec%%:*}"; m="${spec#*:}"
  uv run python evaluate_summarisation.py --model "$m" \
    --dataset "$DATASETS_DIR/xsum_dataset/test" --limit 300 \
    --output "$RESULTS_DIR/qwen3-${n}-xsum.json"
done
echo "=== classification ==="
for spec in "moe:$MOE_OUT" "xsum-expert:$XSUM_FULL" "crime-expert:$CRIME_FULL"; do
  n="${spec%%:*}"; m="${spec#*:}"
  uv run python evaluate.py --model "$m" \
    --dataset "$DATASETS_DIR/crime_dataset/test" --limit 300 \
    --output "$RESULTS_DIR/qwen3-${n}-crime.json"
done

echo "=== done. MoE at $MOE_OUT (node-local; copy out if it is worth keeping) ==="
