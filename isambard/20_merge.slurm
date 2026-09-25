#!/bin/bash
#SBATCH --job-name=xsum-merge
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge-%A_%a.out
#SBATCH --error=logs/merge-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Every merge the arm needs, one per array task: the N-way merge and the four
# disjoint pair merges. They share no state, so running them serially in one
# job only lengthened the critical path.
#
#   task 1     linear merge of all 8 shard experts          (1/1 of the data)
#   tasks 2-5  linear merge of shards (1,2) (3,4) (5,6) (7,8)  (1/4 each)
#
# Submit after the shard array succeeds:
#   sbatch --dependency=afterok:<shard-array-id> isambard/20_merge.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs

SPLITS="${SPLITS:-8}"
i="$SLURM_ARRAY_TASK_ID"

if [ "$i" = "1" ]; then
  CONFIG="merge_linear_xsum_${SPLITS}way_config.yaml"
  OUT="$MODELS_DIR/gemma3-xsum-merged-linear-${SPLITS}way"
else
  # tasks 2..5 -> pairs (1,2) (3,4) (5,6) (7,8)
  a=$(( (i - 2) * 2 + 1 )); b=$(( a + 1 ))
  CONFIG="merge_linear_xsum_pair${a}${b}_config.yaml"
  OUT="$MODELS_DIR/gemma3-xsum-merged-pair${a}${b}"
fi

echo "=== $CONFIG -> $OUT | $(hostname) | $(date -Is) ==="
# config.json, not the directory: a run that dies inside mergekit leaves an
# empty $OUT behind and a -d test then reports it as already merged.
if [ -f "$OUT/config.json" ]; then echo "already merged, skipping"; exit 0; fi
rm -rf "$OUT"   # clear any partial directory before rebuilding

cd "$REPO_ROOT/merge"
uv run mergekit-yaml "$CONFIG" "$OUT" \
  --cuda --lazy-unpickle --lora-merge-cache "$REPO_ROOT/.lora-cache"

echo "=== done $(date -Is) ==="
