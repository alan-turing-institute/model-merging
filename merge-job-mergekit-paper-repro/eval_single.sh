#!/bin/bash
set -euo pipefail
RESULTS="$1"
NAME="$2"
MODEL_PATH="$3"
export HF_TOKEN="${4:-}"
mkdir -p "$RESULTS/lm-eval"

TASKS="medqa_4options,medmcqa,pubmedqa,arc_challenge,hellaswag,mmlu"

echo "=== Disk layout ===" | tee "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"

echo "=== Evaluating: $NAME ($MODEL_PATH) ===" | tee -a "$RESULTS/run.log"
lm_eval --model hf --model_args pretrained="$MODEL_PATH",dtype=float16 \
  --tasks "$TASKS" --device cuda --batch_size 8 \
  --output_path "$RESULTS/lm-eval/$NAME" \
  --log_samples \
  2>&1 | tee "$RESULTS/lm-eval/$NAME.log"

echo "=== Final disk layout ===" | tee -a "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"
echo "EVAL_COMPLETE"
