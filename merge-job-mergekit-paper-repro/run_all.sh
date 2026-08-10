#!/bin/bash
set -euo pipefail
RESULTS="$1"
MERGED_ROOT="$2"
mkdir -p "$RESULTS/lm-eval"

TASKS="medqa_4options,medmcqa,pubmedqa,arc_challenge,hellaswag,mmlu"

echo "=== Disk layout ===" | tee "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"

run_merge() {
  local method_name="$1"
  local config="$2"
  echo "=== Merging: $method_name ===" | tee -a "$RESULTS/run.log"
  mergekit-yaml "$config" "$MERGED_ROOT/$method_name" --cuda --lazy-unpickle --allow-crimes 2>&1 | tee "$RESULTS/merge-$method_name.log"
  echo "=== Merge done: $method_name ===" | tee -a "$RESULTS/run.log"
  df -h | tee -a "$RESULTS/run.log"
}

run_eval() {
  local name="$1"
  local path="$2"
  echo "=== Evaluating: $name ($path) ===" | tee -a "$RESULTS/run.log"
  lm_eval --model hf --model_args pretrained="$path",dtype=float16 \
    --tasks "$TASKS" --device cuda --batch_size 8 \
    --output_path "$RESULTS/lm-eval/$name" \
    --log_samples \
    2>&1 | tee "$RESULTS/lm-eval/$name.log"
  echo "=== Done: $name ===" | tee -a "$RESULTS/run.log"
}

# Merges first -- this is what populates the shared HF cache with both
# source checkpoints (+ plain Llama-2-7b for TIES/DARE-TIES's common-ancestor
# base_model), so later merges and the baseline evals reuse that cache with
# no re-download.
run_merge "lerp" configs/lerp-2way.yml
run_merge "slerp" configs/slerp-2way.yml
run_merge "ties" configs/ties-2way.yml
run_merge "dare-ties" configs/dare-ties-2way.yml

run_eval "llama2-7b-chat" "NousResearch/Llama-2-7b-chat-hf"
run_eval "meditron-7b" "epfl-llm/meditron-7b"
run_eval "merged-lerp" "$MERGED_ROOT/lerp"
run_eval "merged-slerp" "$MERGED_ROOT/slerp"
run_eval "merged-ties" "$MERGED_ROOT/ties"
run_eval "merged-dare-ties" "$MERGED_ROOT/dare-ties"

echo "=== Final disk layout ===" | tee -a "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"
echo "ALL_RUNS_COMPLETE"
