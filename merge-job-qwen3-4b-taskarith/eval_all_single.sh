#!/bin/bash
set -euo pipefail
RESULTS="$1"
shift

NAMES=(base jan-nano abliterated pythagoras-prover qvikhr-instruction chinese-error-corrector met-d osim merged-linear merged-ties merged-dare-ties merged-task-arithmetic merged-arcee-fusion)
PATHS=("$@")

mkdir -p "$RESULTS/lm-eval"
echo "=== Disk layout ===" | tee "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"

i=0
while [ $i -lt ${#NAMES[@]} ]; do
  name="${NAMES[$i]}"
  path="${PATHS[$i]}"
  echo "=== Evaluating: $name ($path) ===" | tee -a "$RESULTS/run.log"
  lm_eval --model hf --model_args pretrained="$path",dtype=bfloat16 \
    --tasks arc_easy,piqa,hellaswag,mmlu --device cuda --batch_size 16 \
    --output_path "$RESULTS/lm-eval/$name" \
    --log_samples \
    2>&1 | tee "$RESULTS/lm-eval/$name.log"
  echo "=== Done: $name ===" | tee -a "$RESULTS/run.log"
  i=$((i + 1))
done

echo "=== Final disk layout ===" | tee -a "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"
echo "ALL_EVALS_COMPLETE"
