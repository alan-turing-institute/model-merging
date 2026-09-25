#!/bin/bash
#SBATCH --job-name=crime-experts
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/crime-expert-%A_%a.out
#SBATCH --error=logs/crime-expert-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Train the three crime experts on Isambard.
#
# WHY HERE. The crime arm carries the project's suspended claim - distillation
# at 0.2/1.0 scoring 0.9707 against a no-teacher control's 0.9601 with ~50x
# lower variance. It is suspended twice over: the teacher was misaligned by one
# position, and the weighting means something different on this task than on
# XSum. Crime targets are a SINGLE token, so N is roughly the batch (4 x 4 = 16)
# rather than XSum's ~165, and 0.2/1.0 was about 3:1 toward the teacher here
# against 33:1 there. The same nominal weights were never the same experiment.
#
# It is runnable without Azure: crime data comes from the public HF dataset
# Binaryy/crime_posts_reddit, so prepare_data.py rebuilds it from scratch. The
# Azure crime adapters are unreachable while the subscription is read-only, and
# the laptop holds only the full-data one - which, on the XSum evidence, would
# be the wrong generation anyway.
#
# Same caveat as the XSum retraining: these will not be bit-identical to the
# Azure artefacts, so absolute figures do not carry across. Compare within this
# platform.
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs

case "$SLURM_ARRAY_TASK_ID" in
  1) CONFIG=crime_gemma.yaml;  OUT=gemma3-crime-full-lora ;;
  2) CONFIG=crime_gemma1.yaml; OUT=gemma3-crime-1-of-2-lora ;;
  3) CONFIG=crime_gemma2.yaml; OUT=gemma3-crime-2-of-2-lora ;;
  *) echo "unexpected array index $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "=== $CONFIG -> $OUT | $(hostname) | $(date -Is) ==="
grep -E '^eager_attention:|^load_in_4bit:|^num_epochs:|^learning_rate:' "$REPO_ROOT/train/$CONFIG"

if [ -f "$MODELS_DIR/$OUT/adapter_model.safetensors" ]; then
  echo "=== $OUT already trained, skipping ==="
  exit 0
fi

cd "$REPO_ROOT/train"
uv run axolotl train "$CONFIG"

# The failure this job exists to avoid is silent: a NaN run still saves an
# adapter, and the only symptom is an evaluation of nothing an hour later. Fail
# here instead, where the log still says why.
# This job runs axolotl directly, so its output goes to the Slurm log, not to
# the .axolotl-*.log that run_kd_pipeline.sh's wrapper tees. The first version
# of this check read that non-existent file: grep failed, the condition was
# false, and the guard passed every time - a silent check for a silent failure.
TRAIN_LOG="${SLURM_SUBMIT_DIR:-$PWD}/logs/crime-expert-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
if [ ! -f "$TRAIN_LOG" ]; then
  echo "cannot find my own Slurm log at $TRAIN_LOG - refusing to claim success" >&2
  exit 1
fi
if grep -qiE "\bnan\b" "$TRAIN_LOG"; then
  echo "NaN seen in training - eager attention did not take. Refusing to mark complete." >&2
  exit 1
fi
echo "=== $OUT done $(date -Is) ==="
