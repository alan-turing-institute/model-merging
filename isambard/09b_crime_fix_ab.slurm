#!/bin/bash
#SBATCH --job-name=crime-fix-ab
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/crime-legacy-%A_%a.out
#SBATCH --error=logs/crime-legacy-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# The pre-fix half of the alignment A/B. Five seeds, paired by construction
# with results-isambard/crime-distilled-s1..s5.json: same merge, same experts,
# same seeds, same kd_alpha 0.0625, same budget, same evaluator, same test
# set. The teacher's row convention is the only difference, so the five
# differences licence a paired test.
#
# Submit after 09a:
#   sbatch --dependency=afterok:<09a jobid> isambard/09b_crime_fix_ab.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

SEED="$SLURM_ARRAY_TASK_ID"
CONFIG="crime_kd_legacy_s${SEED}.yaml"
STUDENT="$MODELS_DIR/gemma3-crime-kd-legacy-s${SEED}"
MERGE="$MODELS_DIR/gemma3-crime-merged-linear-2"

echo "=== crime legacy-teacher seed $SEED | $(hostname) | $(date -Is) ==="
grep -E '^seed:|^kd_alpha:|^kd_ce_alpha:|^output_dir:' "$REPO_ROOT/train/$CONFIG"
grep -E 'crime_kd_train' "$REPO_ROOT/train/$CONFIG"

[ -f "$MERGE/config.json" ] || { echo "merge missing at $MERGE" >&2; exit 1; }
# Gate on the LEGACY marker specifically. Gating on existence is how an unmarked
# directory got trained on before; gating on .kd_alignment_v2 here would silently
# turn this into a duplicate of the post-fix arm, which is the one failure mode
# that would look like a clean null result.
[ -f "$DATASETS_DIR/crime_kd_train_legacy/.kd_alignment_legacy" ] || {
  echo "pre-fix teacher data missing or unmarked at $DATASETS_DIR/crime_kd_train_legacy" >&2
  echo "run isambard/09a_crime_legacy_teacher.slurm first" >&2; exit 1; }
[ ! -f "$DATASETS_DIR/crime_kd_train_legacy/.kd_alignment_v2" ] || {
  echo "that directory carries the ALIGNED marker too - refusing" >&2; exit 1; }

if [ -f "$STUDENT/adapter_model.safetensors" ] \
   && [ "$STUDENT/adapter_model.safetensors" -nt "$MERGE/config.json" ]; then
  echo "=== already trained against this merge, skipping ==="
else
  cd "$REPO_ROOT/train"
  uv run axolotl train "$CONFIG"
  cd "$REPO_ROOT"
fi

cd "$REPO_ROOT/evaluate"
uv run python evaluate.py \
  --model "$MERGE" --adapter "$STUDENT" \
  --dataset "$DATASETS_DIR/crime_dataset/test" \
  --output "$RESULTS_DIR/crime-legacy-s${SEED}.json"

echo "=== crime legacy seed $SEED done $(date -Is) ==="
