#!/bin/bash
#SBATCH --job-name=crime-half-seeds
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Score the ten per-seed half-experts that were trained for the method-seed study
# and then never evaluated on their own.
#
# TWO QUESTIONS DEPEND ON THIS, and neither could be answered without it.
#
# How much does an EXPERT move between training runs? The report quotes 0.0039,
# but that is the standard deviation of the MERGE across expert draws, and
# merging averages away much of what varies in its inputs. The four-way arm
# measured the experts themselves for the first time and found 0.0049 to 0.0319
# depending on the shard - one of the four quarters ranged 0.0724 across seeds on
# identical data. If half-experts vary like that, the 0.0288 shift between the
# Azure and Isambard generations is ordinary training noise rather than evidence
# about platforms, and Section 4.1 is attributing it to the wrong cause.
#
# And does a two-way merge beat the best of its own two shards? The five merges
# are already scored; what was missing is the per-seed shards to compare them
# against, which is why that comparison has rested on a single pair.
#
# No training: the adapters are on disk, this is ten evaluations.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
i="$SLURM_ARRAY_TASK_ID"
HALF=$(( (i - 1) / 5 + 1 ))
SEED=$(( (i - 1) % 5 + 1 ))
ADAPTER="$MODELS_DIR/gemma3-crime-${HALF}-of-2-seed${SEED}-lora"
RESULT="$RESULTS_DIR/crime-${HALF}-of-2-seed${SEED}.json"

echo "=== half $HALF, seed $SEED | $(hostname) | $(date -Is) ==="
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "no adapter at $ADAPTER" >&2; exit 1; }
if [ -f "$RESULT" ]; then echo "=== already scored, skipping ==="; exit 0; fi
cd "$REPO_ROOT/evaluate"
uv run python evaluate.py --model google/gemma-3-4b-it --adapter "$ADAPTER" \
  --dataset "$DATASETS_DIR/crime_dataset/test" --output "$RESULT"
[ -f "$RESULT" ] || { echo "no result written" >&2; exit 1; }
echo "=== done $(date -Is) ==="
