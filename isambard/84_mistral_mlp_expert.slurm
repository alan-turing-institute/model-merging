#!/bin/bash
#SBATCH --job-name=mistral-mlp-expert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/mistral-mlp-expert-%j.out
#SBATCH --error=logs/mistral-mlp-expert-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Usage, from the repository root:
#   sbatch isambard/84_mistral_mlp_expert.slurm crime|xsum
#
# One Mistral expert with LoRA on the MLP (train/<task>_mistral_mlp.yaml), then
# the check 80_mistral_smoke and 81_mistral_xsum ran on the attention-only
# experts: does it emit the label, or one sentence, and stop.
#
# Why retrain: mergekit-moe keeps each expert's MLP and takes everything else
# from the base model. The attention-only experts left both MoE experts equal
# to the base MLP, so 82_mistral_moe's MoE was the untuned model - all 200 of
# its crime predictions are identical to 83_mistral_base's.
#
# Exits non-zero, so that 85_mistral_mlp_moe.slurm submitted with
# --dependency=afterok does not run, if the adapter touches anything but the
# MLP or the expert does not clearly beat the untuned model (83_mistral_base:
# crime accuracy 0.765, XSum ROUGE-1 0.272 at 32 words). An expert no better
# than the base cannot show what a merge keeps.
set -euo pipefail
TASK=${1:-}
case "$TASK" in
  crime|xsum) ;;
  *) echo "usage: sbatch isambard/84_mistral_mlp_expert.slurm crime|xsum" >&2; exit 1 ;;
esac
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# The checkout this was submitted from, not config.sh's default of
# ~/model-merging, which may be on another branch without these files.
export REPO_ROOT="${REPO_ROOT:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

OUT="$MODELS_DIR/mistral-$TASK-mlp-lora"
# Resolve the snapshot directly, as 80_mistral_smoke does: the cached snapshot
# omits consolidated.safetensors, which snapshot_download's check rejects.
MISTRAL_DIR=$(ls -d "$HF_HUB_CACHE"/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/ | head -1)
MISTRAL_DIR="${MISTRAL_DIR%/}"
echo "base: $MISTRAL_DIR"
[ -f "$MISTRAL_DIR/config.json" ] || { echo "Mistral snapshot not found" >&2; exit 1; }
echo "=== $(hostname) | $(date -Is) | $TASK ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || echo "  (nvidia-smi unavailable on this node)"

if [ -f "$OUT/adapter_model.safetensors" ]; then
  echo "=== expert already trained: $OUT ==="
else
  # The config with dataset_prepared_path appended, since axolotl does not
  # expand variables in the file. Run from train/, where its relative paths
  # resolve.
  CONFIG="$TMPDIR/${TASK}_mistral_mlp-${SLURM_JOB_ID:-local}.yaml"
  { cat "$REPO_ROOT/train/${TASK}_mistral_mlp.yaml"
    echo "dataset_prepared_path: $MM_PREPARED/mistral-$TASK-mlp"; } > "$CONFIG"
  cd "$REPO_ROOT/train"
  uv run axolotl train "$CONFIG"
  cd "$REPO_ROOT"
fi

TRAIN_LOG="${SLURM_SUBMIT_DIR:-$PWD}/logs/mistral-mlp-expert-${SLURM_JOB_ID:-local}.out"
if [ -f "$TRAIN_LOG" ] && grep -qiE "\bnan\b" "$TRAIN_LOG"; then
  echo "NaN during training - refusing to report this expert as usable" >&2; exit 1
fi

# The point of the retraining: nothing but the MLP.
python3 - "$OUT/adapter_config.json" <<'PY'
import json, sys
targets = set(json.load(open(sys.argv[1]))["target_modules"])
print("adapter target_modules:", sorted(targets))
if targets != {"gate_proj", "up_proj", "down_proj"}:
    sys.exit("the adapter is not MLP-only")
PY

# One evaluation, with its full output kept in logs/ and its metric lines
# printed. The output JSON is removed first, so that a crash cannot leave an
# earlier run's result to be read below as this one's.
run_eval() {  # <output json> <metric regex> <command ...>
  local out=$1 pattern=$2 log
  shift 2
  log="${SLURM_SUBMIT_DIR:-$PWD}/logs/$(basename "$out" .json)-${SLURM_JOB_ID:-local}.log"
  rm -f "$out"
  if "$@" --output "$out" > "$log" 2>&1; then
    grep -E "$pattern" "$log" || true
  else
    echo "!!! evaluation failed (exit $?); the end of $log:"
    tail -20 "$log"
  fi
}

cd "$REPO_ROOT/evaluate"
if [ "$TASK" = crime ]; then
  # Does it emit the label, and stop? Three budgets, as 80_mistral_smoke.
  for T in 5 16 48; do
    echo "=== generation budget $T ==="
    run_eval "$RESULTS_DIR/mistral-mlp-crime-t${T}.json" "^Accuracy|^Unparsed" \
      uv run python evaluate.py --model "$MISTRAL_DIR" \
      --adapter "$OUT" --dataset "$DATASETS_DIR/crime_dataset/test" \
      --limit 80 --max-new-tokens $T
  done
else
  # Does it write one sentence, and stop? As 81_mistral_xsum: a model that
  # never emits end-of-turn generates to the 64-token cap.
  run_eval "$RESULTS_DIR/mistral-mlp-xsum-full.json" "^rouge1|^semantic|Length|^Examples" \
    uv run python evaluate_summarisation.py --model "$MISTRAL_DIR" \
    --adapter "$OUT" --dataset "$DATASETS_DIR/xsum_dataset/test" --limit 200
fi

cd "$REPO_ROOT"
python3 - "$TASK" <<'PY'
import json, os, sys

task, results = sys.argv[1], os.environ["RESULTS_DIR"]
if task == "crime":
    print("\nbudget  accuracy  unparsed  first prediction")
    for t in (5, 16, 48):
        f = os.path.join(results, "mistral-mlp-crime-t%d.json" % t)
        if not os.path.exists(f):
            print("%-7d (missing)" % t)
            continue
        d = json.load(open(f))
        m = d["metrics"]
        print("%-7d %-9.4f %-9s %r" % (t, m["accuracy"], m["num_unparsed"],
                                       str(d.get("predictions", [""])[0])[:50]))
    m = json.load(open(os.path.join(results, "mistral-mlp-crime-t16.json")))["metrics"]
    ok = m["accuracy"] >= 0.85 and m["num_unparsed"] <= 4
    print("\ngate at 16 tokens: accuracy %.4f >= 0.85 and unparsed %d <= 4 of 80: %s"
          % (m["accuracy"], m["num_unparsed"], "pass" if ok else "FAIL"))
    print("Untuned Mistral: 0.765 on the first 200 (83_mistral_base); the")
    print("attention-only expert: 0.975 at five tokens. ~0.44 is the always-positive rate.")
else:
    d = json.load(open(os.path.join(results, "mistral-mlp-xsum-full.json")))
    m = d["metrics"]
    print("\nROUGE-1 %.4f   words %.1f   sentences %.2f   empty %d" % (
        m["rouge1"], m["mean_pred_words"], m["mean_pred_sentences"], m["num_empty"]))
    print("first prediction: %r" % str(d.get("predictions", [""])[0])[:110])
    ok = m["rouge1"] >= 0.35 and m["mean_pred_words"] <= 35
    print("\ngate: ROUGE-1 %.4f >= 0.35 and %.1f words <= 35: %s"
          % (m["rouge1"], m["mean_pred_words"], "pass" if ok else "FAIL"))
    print("Untuned Mistral: 0.272 at 32.0 words (83_mistral_base); the attention-only")
    print("expert: 0.453 at 19.1. References are 21.0 words; near the cap means no stop.")
sys.exit(0 if ok else 1)
PY
echo "=== $TASK MLP expert complete $(date -Is) ==="
