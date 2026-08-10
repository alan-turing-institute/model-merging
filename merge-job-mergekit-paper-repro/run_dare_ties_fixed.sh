#!/bin/bash
set -euo pipefail
RESULTS="$1"
MERGED_ROOT="$2"
export HF_TOKEN="$3"
mkdir -p "$RESULTS/lm-eval"

TASKS="medqa_4options,medmcqa,pubmedqa,arc_challenge,hellaswag,mmlu"

echo "=== Disk layout ===" | tee "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"

# Root cause of the earlier 100%-NaN corruption: dare_ties defaults to
# rescale=True (the "REscale" in DARE), which divides by the L1 norm of
# the surviving (post-Bernoulli-mask) elements of the delta tensor. For
# embed_tokens/lm_head specifically, a random mask draw can leave a tiny
# surviving norm, producing a huge rescale factor that overflows fp16's
# ~65504 max range to Inf, which then propagates to NaN. Switching dtype
# to bfloat16 (~3e38 max, same exponent range as fp32) removes the
# overflow risk while preserving DARE's actual rescale behavior --
# unlike disabling rescale entirely, which would no longer be DARE-TIES.
echo "=== Merging: dare-ties (bfloat16 fix) ===" | tee -a "$RESULTS/run.log"
mergekit-yaml configs/dare-ties-2way.yml "$MERGED_ROOT/dare-ties-fixed" --cuda --lazy-unpickle --allow-crimes 2>&1 | tee "$RESULTS/merge-dare-ties-fixed.log"
echo "=== Merge done: dare-ties-fixed ===" | tee -a "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"

echo "=== Evaluating: merged-dare-ties-fixed ===" | tee -a "$RESULTS/run.log"
lm_eval --model hf --model_args pretrained="$MERGED_ROOT/dare-ties-fixed",dtype=bfloat16 \
  --tasks "$TASKS" --device cuda --batch_size 8 \
  --output_path "$RESULTS/lm-eval/merged-dare-ties-fixed" \
  --log_samples \
  2>&1 | tee "$RESULTS/lm-eval-dare-ties-fixed.log"

echo "=== Final disk layout ===" | tee -a "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"
echo "DARE_TIES_FIX_COMPLETE"
