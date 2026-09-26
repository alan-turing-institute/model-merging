#!/bin/bash
#SBATCH --job-name=crime-legacy-teacher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/crime-legacy-teacher-%j.out
#SBATCH --error=logs/crime-legacy-teacher-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Prep for the controlled A/B of the alignment fix 6f923a1: regenerate the
# crime teacher logprobs in the PRE-FIX row convention, holding the experts,
# the merge and the data fixed.
#
# WHY. The fix's effect on the end task has only ever been read across hardware
# generations - the pre-fix five seeds ran on Azure v1 experts, the post-fix
# five on Isambard v2 - and the retraining is a confound large enough to
# produce the whole inversion on its own. This arm changes one thing.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

MERGE="$MODELS_DIR/gemma3-crime-merged-linear-2"

echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || echo '  (nvidia-smi unavailable on this node)'

[ -f "$MERGE/config.json" ] || { echo "merge missing at $MERGE" >&2; exit 1; }
# The comparator must already be on disk, or there is no A to the B.
for i in 1 2 3 4 5; do
  [ -f "$RESULTS_DIR/crime-distilled-s$i.json" ] \
    || { echo "post-fix arm incomplete: crime-distilled-s$i.json missing" >&2; exit 1; }
done

# --- 0. the gate, inverted -------------------------------------------------
# 07_kd_crime.slurm refuses to run unless the teacher==student KL is 0.0000.
# This job needs a teacher that is NOT aligned, so it asserts both halves: the
# fixed convention must still be ~0 (the producer works) AND the old convention
# must still be far from it (the bug is still reproducible). If the old
# convention had drifted to ~0 the two arms would be the same experiment.
echo "=== alignment smoke: fixed must be 0.0000, old must not be ==="
uv run --project evaluate python experiments/kd_alignment_smoke.py --examples 4 \
  --dataset "$DATASETS_DIR/crime_dataset/train" \
  | tee "$RESULTS_DIR/alignment_check_legacy.txt"
grep -qE 'one extra row \(6f923a1\).*[[:space:]]0\.0000$' "$RESULTS_DIR/alignment_check_legacy.txt" \
  || { echo "fixed convention is not 0.0000 - the producer is wrong" >&2; exit 1; }
grep -qE 'one row per assistant token.*[[:space:]][1-9][0-9]*\.[0-9]{4}$' "$RESULTS_DIR/alignment_check_legacy.txt" \
  || { echo "old convention no longer measures misaligned - nothing to A/B" >&2; exit 1; }

# --- 1. teacher logprobs, pre-fix convention -------------------------------
for i in 1 2; do
  OUT="$DATASETS_DIR/crime_kd_legacy_half$i"
  if [ -f "$OUT/.kd_alignment_legacy" ]; then
    echo "=== legacy teacher $i already at $OUT ==="
    continue
  fi
  rm -rf "$OUT"
  echo "=== legacy teacher $i logprobs ==="
  uv run --project evaluate python precompute_logprobs.py \
    --model google/gemma-3-4b-it \
    --adapter "$MODELS_DIR/gemma3-crime-$i-of-2-lora" \
    --dataset "$DATASETS_DIR/crime_dataset$i/train" \
    --output "$OUT" --verify 32 --legacy-alignment
done

# Marker, not directory existence - the fourth bug of that shape in this
# pipeline was a concat that wrote its output and died before stamping it.
if [ ! -f "$DATASETS_DIR/crime_kd_train_legacy/.kd_alignment_legacy" ]; then
  rm -rf "$DATASETS_DIR/crime_kd_train_legacy"
  echo "=== combining the two legacy halves ==="
  uv run --project evaluate python concat_datasets.py \
    "$DATASETS_DIR/crime_kd_legacy_half1" "$DATASETS_DIR/crime_kd_legacy_half2" \
    --output "$DATASETS_DIR/crime_kd_train_legacy"
fi

# --- 2. the two teacher sets must differ in row count, and nowhere else ----
# Cheap, and it is the one assertion that distinguishes this A/B from a
# five-seed re-run of the same thing.
uv run --project evaluate python - <<'PY'
import os
from datasets import load_from_disk
d = os.environ["DATASETS_DIR"]
new = load_from_disk(f"{d}/crime_kd_train")
old = load_from_disk(f"{d}/crime_kd_train_legacy")
assert len(new) == len(old), f"example counts differ: {len(new)} vs {len(old)}"
deltas = {len(a["logprobs"]) - len(b["logprobs"])
          for a, b in zip(new.select(range(64)), old.select(range(64)))}
print(f"{len(new)} examples; row-count delta (aligned - legacy): {deltas}")
assert deltas == {1}, f"expected every aligned example to carry exactly one extra row, got {deltas}"
print("OK: the two teacher sets differ by exactly the filler row.")
PY

echo "=== 3. generating the five pre-fix configs ==="
python3 isambard/make_crime_legacy_seeds.py

echo "=== legacy teacher ready $(date -Is) ==="
