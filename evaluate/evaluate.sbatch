#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.out
# Usage, from evaluate/:
#   sbatch evaluate.sbatch <task> <model> [--adapter <dir>] [inspect eval args...]
#   e.g. sbatch evaluate.sbatch xsum google/gemma-3-4b-it -T prompt_variant=short
#        sbatch evaluate.sbatch xsum ../models/xsum/gemma3_4b_xsum_short \
#            -T prompt_variant=short --limit 50
#        sbatch evaluate.sbatch crime google/gemma-3-4b-it \
#            --adapter ../models/crime/gemma3_4b_crime
#        sbatch evaluate.sbatch translation google/gemma-3-4b-it \
#            -T prompt_variant=short -T split=validation
#        sbatch evaluate.sbatch mt_domain_de_en google/gemma-3-1b-it \
#            --adapter ../models/mt_domain_de_en_1b/seed_1/gemma3_1b_mt_domain_de_en
#
# task is crime, wine, clinc150, xsum, translation, translation_en_zh or mt_domain_de_en (runs <task>_task.py). model is a Hugging Face hub id or
# a full model saved on disk (config.json + model.safetensors); --adapter puts a
# PEFT/LoRA adapter on top of it via the hf-peft provider. Everything after that
# goes to `inspect eval` unchanged. xsum and the translation tasks need
# -T prompt_variant=short|long|teacher, and clinc150 -T prompt_variant=short|labels,
# matching the prompt the model was trained on, so it is required here;
# mt_domain_de_en defaults to the prompt it is trained on (prompt_variant=trained). For translation, COMET is a second step:
# sbatch comet_score.sbatch on the resulting log.
#
# The .eval log is written to logs/; read it with `uv run inspect view`. Its
# metadata says what the model is: for a model or adapter dir with a
# provenance.json, its kind, method, base model, training data, teacher and
# targets, coverage, hyperparameters and final losses (see ../provenance.py);
# for a Hub id, model_kind=base. eval_slurm_job links it to slurm/<job>.out.
# A model whose record says its run went NaN (health.ok false) is refused, as
# it can only score NaN; set ALLOW_BROKEN_MODEL=1 to evaluate it anyway.
set -euo pipefail
USAGE="usage: sbatch evaluate.sbatch <crime|wine|clinc150|xsum|translation|translation_en_zh|mt_domain_de_en> <model> [--adapter <dir>] [inspect eval args...]"
TASK=${1:?$USAGE}
MODEL=${2:?$USAGE}
shift 2

ADAPTER=
if [[ ${1:-} == --adapter ]]; then
  ADAPTER=${2:?$USAGE}
  shift 2
fi

# Submit from evaluate/: the task files import semantic_metrics from here, and
# relative model paths resolve against it.
cd "$SLURM_SUBMIT_DIR"
TASK_FILE=${TASK}_task.py
[[ -f $TASK_FILE ]] || { echo "no $TASK_FILE in $PWD - submit from evaluate/" >&2; exit 1; }

if [[ $TASK != crime && $TASK != wine && $TASK != mt_domain_de_en && " $* " != *" prompt_variant="* ]]; then
  [[ $TASK == clinc150 ]] && variants=short\|labels || variants=short\|long\|teacher
  echo "$TASK needs -T prompt_variant=$variants (the prompt the model was trained on)" >&2
  exit 1
fi

# A directory on disk is loaded through model_path; the name after hf/ is then
# only a label in the log.
MODEL_ARGS=()
if [[ -d $MODEL ]]; then
  MODEL_ARGS+=(-M "model_path=$MODEL")
  MODEL_NAME=$(basename "$MODEL")
else
  MODEL_NAME=$MODEL
fi
if [[ -n $ADAPTER ]]; then
  PROVIDER=hf-peft
  MODEL_ARGS+=(-M "adapter_path=$ADAPTER")
else
  PROVIDER=hf
fi

# The nodes' driver (580.173) now supports the cu130 wheels natively, and the
# older CUDA 13.0 forward-compat libcuda (580.65) breaks it with error 803. sbatch
# copies the submitting shell's LD_LIBRARY_PATH, so strip the compat dir from it.
LD_LIBRARY_PATH=$(tr ':' '
' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH

# Hugging Face downloads (base models, the semantic scorer's encoder) live in
# the project area, not the 100 GB home quota. See the main README.
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is a CUDA compat lib on LD_LIBRARY_PATH?'"

# With --adapter, the adapter is the trained model, so describe that.
METADATA=(--metadata "eval_slurm_job=$SLURM_JOB_ID")
DESCRIPTION=$(uv run python ../provenance.py describe "${ADAPTER:-$MODEL}")
while IFS= read -r line; do
  METADATA+=(--metadata "$line")
done <<<"$DESCRIPTION"
if grep -qx 'model_ok=false' <<<"$DESCRIPTION" && [[ ${ALLOW_BROKEN_MODEL:-} != 1 ]]; then
  echo "${ADAPTER:-$MODEL} is broken (NaN in training or its weights; see health in its" \
    "provenance.json). Set ALLOW_BROKEN_MODEL=1 to evaluate it anyway." >&2
  exit 1
fi

# Greedy decoding and bf16 weights for every model, whatever its config.json
# says: Inspect's hf provider defaults do_sample to true, and a checkpoint whose
# config says float32 would otherwise load in fp32 next to a bf16 base model.
uv run inspect eval "$TASK_FILE" --model "$PROVIDER/$MODEL_NAME" "${MODEL_ARGS[@]}" \
  -M batch_size=8 -M do_sample=false -M dtype=bfloat16 \
  "${METADATA[@]}" --log-dir logs "$@"
