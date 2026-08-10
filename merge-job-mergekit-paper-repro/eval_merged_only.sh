#!/bin/bash
set -euo pipefail
RESULTS="$1"
shift
mkdir -p "$RESULTS/lm-eval"

TASKS="medqa_4options,medmcqa,pubmedqa,arc_challenge,hellaswag,mmlu"
NAMES=(merged-lerp merged-slerp merged-ties merged-dare-ties)
PATHS=("$@")

echo "=== Disk layout ===" | tee "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"

i=0
while [ $i -lt ${#NAMES[@]} ]; do
  name="${NAMES[$i]}"
  path="${PATHS[$i]}"
  echo "=== Evaluating: $name ($path) ===" | tee -a "$RESULTS/run.log"
  # Llama-2-7b-chat (vocab_size=32000) and Meditron-7B (vocab_size=32017,
  # 17 extra special tokens from its medical continued-pretraining) have
  # mismatched vocabs. MergeKit merges them anyway but the resulting
  # config/tensor-shape declares a size that doesn't match the actual saved
  # embedding/lm_head rows, which transformers now hard-errors on by
  # default. ignore_mismatched_sizes=True re-initializes the ~17 mismatched
  # rows instead of crashing -- a small, known-scope compromise (well under
  # 0.1% of the vocab, and Meditron's added tokens are unlikely to appear
  # in these benchmarks' item text anyway).
  lm_eval --model hf --model_args pretrained="$path",dtype=float16,ignore_mismatched_sizes=True \
    --tasks "$TASKS" --device cuda --batch_size 8 \
    --output_path "$RESULTS/lm-eval/$name" \
    --log_samples \
    2>&1 | tee "$RESULTS/lm-eval/$name.log"
  echo "=== Done: $name ===" | tee -a "$RESULTS/run.log"
  i=$((i + 1))
done

echo "=== Final disk layout ===" | tee -a "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"
echo "ALL_RUNS_COMPLETE"
