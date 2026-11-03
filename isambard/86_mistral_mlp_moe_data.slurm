#!/bin/bash
#SBATCH --job-name=mistral-mlp-moe-data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/mistral-mlp-moe-data-%j.out
#SBATCH --error=logs/mistral-mlp-moe-data-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Usage, from the repository root, once both 84_mistral_mlp_expert.slurm
# adapters exist:
#   sbatch isambard/86_mistral_mlp_moe_data.slurm
#
# 85_mistral_mlp_moe's MoE with routing prompts cut from the inputs themselves
# (merge/make_mistral_data_router_config.py), on the same two MLP-LoRA experts.
# 85's instruction prompts sent 25% of XSum prompt tokens to the XSum expert
# (38% with negatives), and the MoE kept 18% (73%) of XSum, against 100% for
# a plain linear merge. Two arms: the data prompts alone ("moe-data"), and
# with the other task's inputs as negatives ("moe-data-negative").
#
# Scored and probed exactly as 85, so its rows are the comparison; the
# generated configs are kept beside the results, since they are the record of
# the prompts. Models are built in project scratch and deleted once scored.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# The checkout this was submitted from, not config.sh's default of
# ~/model-merging, which may be on another branch without these files.
export REPO_ROOT="${REPO_ROOT:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

BASE=$(ls -d "$HF_HUB_CACHE"/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/ | head -1)
BASE="${BASE%/}"
# Project space, not $TMPDIR: /tmp on these nodes is tmpfs, so the models
# there would be held in memory, against the job's 115 GB.
WORK="${PROJECTDIR:-/projects/u6ui}/$USER/mm/tmp/mistral-mlp-moe-data-${SLURM_JOB_ID:-local}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
XS="$WORK/xsum-expert"; CR="$WORK/crime-expert"

echo "=== $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no nvidia-smi)"
df -h "$WORK" | tail -1
echo "base: $BASE"
for t in crime xsum; do
  [ -f "$MODELS_DIR/mistral-$t-mlp-lora/adapter_model.safetensors" ] \
    || { echo "no $t MLP adapter - run 84_mistral_mlp_expert.slurm $t first" >&2; exit 1; }
done

# One evaluation, with its full output kept in logs/ and its metric lines
# printed. The output JSON is removed first, so that a crash cannot leave an
# earlier run's result in the summary as this one's.
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

score() {  # <name> <model dir>: 82's protocol, both tasks
  local name=$1 model=$2
  cd "$REPO_ROOT/evaluate"
  echo "=== $name on crime ==="
  run_eval "$RESULTS_DIR/mistral-mlp-${name}-crime.json" "^Accuracy|^Unparsed" \
    uv run python evaluate.py --model "$model" \
    --dataset "$DATASETS_DIR/crime_dataset/test" --limit 200 --max-new-tokens 16
  echo "=== $name on xsum ==="
  run_eval "$RESULTS_DIR/mistral-mlp-${name}-xsum.json" "^rouge1|Length" \
    uv run python evaluate_summarisation.py --model "$model" \
    --dataset "$DATASETS_DIR/xsum_dataset/test" --limit 200
  cd "$REPO_ROOT"
}

# --- the parents, materialised as in 85 -----------------------------------
for spec in "xsum:$XS" "crime:$CR"; do
  t="${spec%%:*}"; out="${spec#*:}"
  echo "=== materialising the $t expert ==="
  sed -e "s|LOCAL_BASE|$BASE|g" -e "s|ADAPTER|$MODELS_DIR/mistral-$t-mlp-lora|g" \
    "$REPO_ROOT/merge/merge_mistral_mlp_materialise.yaml" > "$WORK/materialise-$t.yaml"
  (cd "$REPO_ROOT/merge" && uv run mergekit-yaml "$WORK/materialise-$t.yaml" "$out" \
     --cuda --lazy-unpickle --lora-merge-cache "$WORK/lora-cache")
done

# --- the MoE with data-like routing prompts, without and with negatives ----
for arm in "moe-data:" "moe-data-negative:--negatives"; do
  name="${arm%%:*}"; out="$WORK/$name"; cfg="$RESULTS_DIR/mistral-mlp-${name}-config.yml"
  echo "=== routing prompts: $name ==="
  # Training rows, not the test rows scored below.
  (cd "$REPO_ROOT/merge" && uv run --project ../evaluate python make_mistral_data_router_config.py \
     --tokenizer "$BASE" --base-model "$BASE" --xsum-expert "$XS" --crime-expert "$CR" \
     --crime "$DATASETS_DIR/crime_dataset/train" --xsum "$DATASETS_DIR/xsum_dataset/train" \
     --n 64 --seed 0 ${arm#*:} --output "$cfg")
  echo "=== mergekit-moe: $name ==="
  (cd "$REPO_ROOT/merge" && uv run mergekit-moe "$cfg" "$out" --cuda --lazy-unpickle --allow-crimes)
  score "$name" "$out"
  echo "=== routing: $name ==="
  (cd "$REPO_ROOT/merge" && uv run --project ../evaluate python mistral_router_probe.py "$out" \
     --crime "$DATASETS_DIR/crime_dataset/test" --xsum "$DATASETS_DIR/xsum_dataset/test" \
     --limit 200 --output "$RESULTS_DIR/mistral-mlp-${name}-routing.json") \
    || echo "!!! routing probe failed for $name - its scores above still stand"
  rm -rf "$out"
done

# --- summary, with 85's rows for comparison --------------------------------
python3 - <<'PY'
import json, os

results = os.environ["RESULTS_DIR"]


def metrics(name):
    f = os.path.join(results, name)
    return json.load(open(f))["metrics"] if os.path.exists(f) else None


key = {"crime": "accuracy", "xsum": "rouge1"}
base = {t: metrics("mistral-base-%s.json" % t) for t in key}
home = {t: metrics("mistral-mlp-%s-expert-%s.json" % (t, t)) for t in key}


def retained(task, m):
    # (merged - untuned) / (own expert - untuned), as in 85.
    if not (m and base[task] and home[task]):
        return "-"
    gain = home[task][key[task]] - base[task][key[task]]
    return "%.3f" % ((m[key[task]] - base[task][key[task]]) / gain) if gain > 0 else "-"


print("\nfirst 200 of each test set; retained = (model - untuned) / (own expert - untuned)")
print("%-18s %9s %9s %6s %9s %9s %12s" % ("model", "crime acc", "xsum R-1", "words",
                                          "ret crime", "ret xsum", "routed home"))
for name in ("xsum-expert", "crime-expert", "linear", "moe", "moe-negative", "moe-data", "moe-data-negative"):
    c, x = (metrics("mistral-mlp-%s-%s.json" % (name, t)) for t in ("crime", "xsum"))
    route = os.path.join(results, "mistral-mlp-%s-routing.json" % name)
    routed = ""
    if os.path.exists(route):
        r = json.load(open(route))
        routed = "%.2f / %.2f" % (r["crime"]["home_share"], r["xsum"]["home_share"])
    print("%-18s %9s %9s %6s %9s %9s %12s" % (
        name, "%.4f" % c["accuracy"] if c else "-", "%.4f" % x["rouge1"] if x else "-",
        "%.1f" % x["mean_pred_words"] if x else "-", retained("crime", c), retained("xsum", x), routed))
print("\nrouted home: share of crime / XSum prompt tokens the router sends to that task's expert.")
PY
echo "=== done $(date -Is) ==="
