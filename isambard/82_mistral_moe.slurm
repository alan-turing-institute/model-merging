#!/bin/bash
#SBATCH --job-name=mistral-moe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=06:00:00
#SBATCH --output=logs/mistral-moe-%j.out
#SBATCH --error=logs/mistral-moe-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# The sparse-MoE arm, on a base mergekit actually supports and with both
# experts verified first.
#
# Comparators already measured on Gemma-3, same tasks: a cross-task merge
# retains 98.0% of each ability, and adapter routing with a perfect router
# reaches the oracle ceiling. This adds the third architecture.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

BASE=$(ls -d "$HF_HUB_CACHE"/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/ | head -1)
BASE="${BASE%/}"
WORK="$TMPDIR/mistral-moe"; mkdir -p "$WORK"
XS="$WORK/xsum-expert"; CR="$WORK/crime-expert"; MOE="$WORK/moe-2way"

echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no nvidia-smi)"
df -h "$TMPDIR" | tail -1
echo "base: $BASE"

# --- materialise ----------------------------------------------------------
for spec in "xsum:$XS" "crime:$CR"; do
  tag="${spec%%:*}"; out="${spec#*:}"
  [ -f "$out/config.json" ] && { echo "=== $tag present ==="; continue; }
  rm -rf "$out"
  echo "=== materialising $tag ==="
  sed "s|LOCAL_BASE|$BASE|" "$REPO_ROOT/merge/merge_mistral_materialise_${tag}.yaml" \
    > "$WORK/mat-$tag.yml"
  (cd "$REPO_ROOT/merge" && uv run mergekit-yaml "$WORK/mat-$tag.yml" "$out" \
     --cuda --lazy-unpickle --lora-merge-cache "$TMPDIR/lora-$SLURM_JOB_ID")
done

# --- build the MoE --------------------------------------------------------
if [ ! -f "$MOE/config.json" ]; then
  rm -rf "$MOE"
  sed -e "s|LOCAL_BASE|$BASE|" -e "s|MATERIALISED_XSUM|$XS|" -e "s|MATERIALISED_CRIME|$CR|" \
    "$REPO_ROOT/merge/moe_mistral_2way.yml" > "$WORK/moe.yml"
  echo "=== mergekit-moe ==="
  (cd "$REPO_ROOT/merge" && uv run mergekit-moe "$WORK/moe.yml" "$MOE" \
     --cuda --lazy-unpickle --allow-crimes)
fi
python3 -c "
import json;c=json.load(open('$MOE/config.json'))
print('MoE:', {k:c.get(k) for k in ('model_type','architectures','num_local_experts','num_experts_per_tok')})"

# --- evaluate: MoE against both parents, both tasks -----------------------
cd "$REPO_ROOT/evaluate"
for spec in "moe:$MOE" "xsum-expert:$XS" "crime-expert:$CR"; do
  n="${spec%%:*}"; m="${spec#*:}"
  echo "=== $n on crime ==="
  uv run python evaluate.py --model "$m" \
    --dataset "$DATASETS_DIR/crime_dataset/test" --limit 200 --max-new-tokens 16 \
    --output "$RESULTS_DIR/mistral-${n}-crime.json" 2>&1 | grep -E "^Accuracy|^Unparsed" || true
  echo "=== $n on xsum ==="
  uv run python evaluate_summarisation.py --model "$m" \
    --dataset "$DATASETS_DIR/xsum_dataset/test" --limit 200 \
    --output "$RESULTS_DIR/mistral-${n}-xsum.json" 2>&1 | grep -E "^rouge1|Length" || true
done
echo "=== done $(date -Is) ==="
