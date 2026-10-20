#!/bin/bash
#SBATCH --job-name=summ-metrics
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# BERTScore and DeepEval's SummarizationMetric on finished evals (summ_metrics.py).
# Usage, from evaluate/:
#   sbatch summ_metrics.sbatch --task xsum --models 'kdgold75|^base' [--limit 20] [--metrics bertscore]
#
# With deepeval in --metrics (the default), the judge is served by vLLM from the
# self-distill environment, on this job's GPU, as an OpenAI-compatible endpoint
# on localhost; JUDGE_MODEL picks it (default: Qwen3-32B, downloaded by
# jobs/dl_qwen3_32b.sh at a pinned revision; bf16, 65 GB). vLLM takes
# JUDGE_GPU_SHARE of the GPU, leaving the rest for BERTScore's encoder. The
# scorer runs in its own script environment (summ_metrics.py.lock); build it once
# on a login node first:  uv sync --script summ_metrics.py
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
# The cache, not HF_HOME: HF_HOME would move the auth token too (see evaluate.sbatch).
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECTDIR/$USER/mm/hf-cache}
export DEEPEVAL_TELEMETRY_OPT_OUT=1

JUDGE=${JUDGE_MODEL:-$PROJECTDIR/$USER/mm/models/qwen3-32b}
JUDGE_LABEL=${JUDGE_LABEL:-$(basename "$JUDGE")}
PORT=$((20000 + SLURM_JOB_ID % 20000))
if [[ " $* " != *" --metrics "* || " $* " == *" deepeval"* || " $* " == *" judge"* ]]; then
  # The nodes have no CUDA toolkit: FlashInfer's sampler JIT-compiles with nvcc,
  # and torch.compile / CUDA graphs did not work either - the settings
  # generate_teacher_logprobs.py runs vLLM with.
  # JUDGE_COMPACT_JSON=1: no free whitespace in schema-constrained output. The
  # grammar otherwise lets the judge emit newlines until max_tokens, which was
  # most of the 2% of XSum judgements still unparsable after the first retry.
  # Opt-in, so a retry does not change how the bulk of a run was decoded.
  EXTRA=()
  [[ -n ${JUDGE_COMPACT_JSON:-} ]] && EXTRA=(--structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}')
  VLLM_USE_FLASHINFER_SAMPLER=0 UV_PROJECT_ENVIRONMENT=${SELF_DISTILL_VENV:-$PROJECTDIR/$USER/mm/venvs/self-distill} \
    uv run --no-sync --project ../self-distill vllm serve "$JUDGE" --served-model-name judge --enforce-eager \
      --host 127.0.0.1 --port "$PORT" --max-model-len "${JUDGE_MAX_LEN:-16384}" \
      --gpu-memory-utilization "${JUDGE_GPU_SHARE:-0.85}" --seed 0 "${EXTRA[@]}" > "judge-$SLURM_JOB_ID.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true' EXIT
  for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "judge server exited - see judge-$SLURM_JOB_ID.log" >&2; exit 1; }
    sleep 10
  done
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "judge server not up after 20 minutes" >&2; exit 1; }
  echo "judge: $JUDGE on port $PORT"
fi
uv run --script summ_metrics.py logs --judge-url "http://127.0.0.1:$PORT/v1" --judge-model judge \
  --judge-label "$JUDGE_LABEL" "$@"
