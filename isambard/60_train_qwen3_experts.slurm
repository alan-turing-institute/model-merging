#!/bin/bash
#SBATCH --job-name=qwen3-experts
#SBATCH --array=1-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/qwen3-expert-%A_%a.out
#SBATCH --error=logs/qwen3-expert-%A_%a.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Train the six Qwen3 experts - both tasks, full and halves.
#
# WHY A SECOND BASE. mergekit-moe emits Mixtral, DeepSeek, Qwen and Qwen3
# architectures and has no Gemma path, so a real sparse-MoE merge of these
# experts cannot be built on gemma-3-4b-it. Qwen3-4B matches it in scale.
#
# NOTHING HERE IS COMPARABLE WITH THE GEMMA-3 RESULTS. Different base,
# tokenizer and pretraining, so every baseline is re-established inside this
# port - which is why the full-data experts are trained alongside the halves
# rather than carried over. Read these six against each other, never against
# 0.4047 / 0.3990 / 0.9694.
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs

case "$SLURM_ARRAY_TASK_ID" in
  1) CONFIG=xsum_qwen3.yaml;   OUT=qwen3-xsum-full-lora ;;
  2) CONFIG=xsum_qwen31.yaml;  OUT=qwen3-xsum-1-of-2-lora ;;
  3) CONFIG=xsum_qwen32.yaml;  OUT=qwen3-xsum-2-of-2-lora ;;
  4) CONFIG=crime_qwen3.yaml;  OUT=qwen3-crime-full-lora ;;
  5) CONFIG=crime_qwen31.yaml; OUT=qwen3-crime-1-of-2-lora ;;
  6) CONFIG=crime_qwen32.yaml; OUT=qwen3-crime-2-of-2-lora ;;
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
TRAIN_LOG="${SLURM_SUBMIT_DIR:-$PWD}/logs/qwen3-expert-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
if [ ! -f "$TRAIN_LOG" ]; then
  echo "cannot find my own Slurm log at $TRAIN_LOG - refusing to claim success" >&2
  exit 1
fi
if grep -qiE "\bnan\b" "$TRAIN_LOG"; then
  echo "NaN seen in training - eager attention did not take. Refusing to mark complete." >&2
  exit 1
fi
echo "=== $OUT done $(date -Is) ==="
