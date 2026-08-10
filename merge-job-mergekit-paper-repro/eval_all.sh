#!/bin/bash
set -euo pipefail
RESULTS="$1"
export HF_TOKEN="$2"
shift 2
mkdir -p "$RESULTS/lm-eval"

TASKS="medqa_4options,medmcqa,pubmedqa,arc_challenge,hellaswag,mmlu"
NAMES=(llama2-7b-chat meditron-7b merged-lerp merged-slerp merged-ties merged-dare-ties)
PATHS=("$@")

echo "=== Disk layout ===" | tee "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"

i=0
while [ $i -lt ${#NAMES[@]} ]; do
  name="${NAMES[$i]}"
  path="${PATHS[$i]}"
  echo "=== Evaluating: $name ($path) ===" | tee -a "$RESULTS/run.log"
  lm_eval --model hf --model_args pretrained="$path",dtype=float16 \
    --tasks "$TASKS" --device cuda --batch_size 8 \
    --output_path "$RESULTS/lm-eval/$name" \
    --log_samples \
    2>&1 | tee "$RESULTS/lm-eval/$name.log"
  echo "=== Done: $name ===" | tee -a "$RESULTS/run.log"
  i=$((i + 1))
done

echo "=== Final disk layout ===" | tee -a "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"
echo "ALL_EVALS_COMPLETE"
