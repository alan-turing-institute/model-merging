#!/bin/bash
set -euo pipefail
RESULTS="$1"
NAME="$2"
MODEL_PATH="$3"
export HF_TOKEN="${4:-}"
# pubmedqa's underlying dataset (bigbio/pubmed_qa) ships a loader script;
# `datasets` requires explicit opt-in to run it (a security default) and
# lm_eval's own trust_remote_code model-arg doesn't propagate to dataset
# loading -- a known open upstream issue (EleutherAI/lm-evaluation-harness
# #2631). This env var is `datasets`' own opt-in mechanism.
export HF_DATASETS_TRUST_REMOTE_CODE="1"
# T4 has 16GB VRAM; a 7B model in float16 alone is already ~14GB, leaving
# little headroom for activation memory. batch_size=8 pushed usage to
# 15.46/15.56 GiB and crashed with CUDA OOM on every single target (base
# and merged alike, since all of them load a 7B model the same way).
# expandable_segments helps reclaim fragmented-but-unallocated memory; it's
# not a fix for a genuine shortfall, but it's a free addition on top of the
# real fix (smaller batch_size).
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$RESULTS/lm-eval"

TASKS="medqa_4options,medmcqa,pubmedqa,arc_challenge,hellaswag,mmlu"

echo "=== Disk layout ===" | tee "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"

echo "=== Evaluating: $NAME ($MODEL_PATH) ===" | tee -a "$RESULTS/run.log"
# Llama-2-7b-chat (vocab_size=32000) and Meditron-7B (vocab_size=32017) have
# mismatched vocabs; merges combining them can carry a config vocab_size
# that doesn't match the merged tensor's actual shape, which transformers
# hard-errors on by default. ignore_mismatched_sizes=True re-initializes the
# handful of mismatched rows instead of crashing -- a no-op for the two base
# (non-merged) checkpoints, which have no mismatch to begin with.
lm_eval --model hf --model_args pretrained="$MODEL_PATH",dtype=float16,ignore_mismatched_sizes=True \
  --tasks "$TASKS" --device cuda --batch_size 2 \
  --output_path "$RESULTS/lm-eval/$NAME" \
  --log_samples \
  2>&1 | tee "$RESULTS/lm-eval/$NAME.log"

echo "=== Final disk layout ===" | tee -a "$RESULTS/run.log"
df -h | tee -a "$RESULTS/run.log"
echo "EVAL_COMPLETE"
