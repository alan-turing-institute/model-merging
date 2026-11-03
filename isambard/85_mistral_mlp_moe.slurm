#!/bin/bash
#SBATCH --job-name=mistral-mlp-moe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/mistral-mlp-moe-%j.out
#SBATCH --error=logs/mistral-mlp-moe-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Usage, from the repository root, after both 84_mistral_mlp_expert.slurm jobs
# have passed (or chained on them with --dependency=afterok:<crime>:<xsum>):
#   sbatch isambard/85_mistral_mlp_moe.slurm
#
# 82_mistral_moe on the MLP-LoRA experts, plus what 82 lacked:
#   - the same MoE with negative prompts (moe_mistral_mlp_2way_negative.yml),
#     which differs only in the router;
#   - two dense merges of the same two experts, task arithmetic and linear, so
#     that the MoE is compared with merging on identical parents;
#   - a routing probe on each MoE (merge/mistral_router_probe.py): the share of
#     each task's prompt tokens the router sends to that task's expert.
# Every model is scored on the first 200 of each test set with 82's protocol
# (crime: 16 new tokens; XSum: the evaluator's default of 64), which is also
# 83_mistral_base's, so that run's results are the untuned reference.
#
# Models are built in project scratch and deleted once scored: each is 14-26
# GB, and all are cheap to rebuild from the two adapters.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# The checkout this was submitted from, not config.sh's default of
# ~/model-merging, which may be on another branch without these files.
export REPO_ROOT="${REPO_ROOT:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"

BASE=$(ls -d "$HF_HUB_CACHE"/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/ | head -1)
BASE="${BASE%/}"
# Project space, not $TMPDIR: /tmp on these nodes is tmpfs, so 55 GB of models
# there would be held in memory, against the job's 115 GB.
WORK="${PROJECTDIR:-/projects/u6ui}/$USER/mm/tmp/mistral-mlp-moe-${SLURM_JOB_ID:-local}"
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

merge_yaml() {  # <config> <output dir>
  (cd "$REPO_ROOT/merge" && uv run mergekit-yaml "$1" "$2" \
     --cuda --lazy-unpickle --lora-merge-cache "$WORK/lora-cache")
}

# --- the parents, materialised: mergekit-moe takes full models only -------
for spec in "xsum:$XS" "crime:$CR"; do
  t="${spec%%:*}"; out="${spec#*:}"
  echo "=== materialising the $t expert ==="
  sed -e "s|LOCAL_BASE|$BASE|g" -e "s|ADAPTER|$MODELS_DIR/mistral-$t-mlp-lora|g" \
    "$REPO_ROOT/merge/merge_mistral_mlp_materialise.yaml" > "$WORK/materialise-$t.yaml"
  merge_yaml "$WORK/materialise-$t.yaml" "$out"
done
score xsum-expert "$XS"
score crime-expert "$CR"

# --- the MoE, with and without negative prompts ---------------------------
for arm in "moe:moe_mistral_mlp_2way.yml" "moe-negative:moe_mistral_mlp_2way_negative.yml"; do
  name="${arm%%:*}"; out="$WORK/$name"
  sed -e "s|LOCAL_BASE|$BASE|g" -e "s|MATERIALISED_XSUM|$XS|g" -e "s|MATERIALISED_CRIME|$CR|g" \
    "$REPO_ROOT/merge/${arm#*:}" > "$WORK/$name.yml"
  echo "=== mergekit-moe: $name ==="
  (cd "$REPO_ROOT/merge" && uv run mergekit-moe "$WORK/$name.yml" "$out" \
     --cuda --lazy-unpickle --allow-crimes)
  python3 -c "
import json; c = json.load(open('$out/config.json'))
print('$name:', {k: c.get(k) for k in ('model_type', 'num_local_experts', 'num_experts_per_tok', 'dtype', 'torch_dtype')})"
  score "$name" "$out"
  echo "=== routing: $name ==="
  (cd "$REPO_ROOT/merge" && uv run --project ../evaluate python mistral_router_probe.py "$out" \
     --crime "$DATASETS_DIR/crime_dataset/test" --xsum "$DATASETS_DIR/xsum_dataset/test" \
     --limit 200 --output "$RESULTS_DIR/mistral-mlp-${name}-routing.json") \
    || echo "!!! routing probe failed for $name - its scores above still stand"
  rm -rf "$out"
done
rm -rf "$XS" "$CR"

# --- dense merges of the same two adapters --------------------------------
for arm in "taskarith:merge_mistral_mlp_crosstask_task_arithmetic.yaml" \
           "linear:merge_mistral_mlp_crosstask_linear.yaml"; do
  name="${arm%%:*}"; out="$WORK/$name"
  sed "s|LOCAL_BASE|$BASE|g" "$REPO_ROOT/merge/${arm#*:}" > "$WORK/$name.yaml"
  echo "=== mergekit-yaml: $name ==="
  merge_yaml "$WORK/$name.yaml" "$out"
  score "$name" "$out"
  rm -rf "$out"
done

# --- summary --------------------------------------------------------------
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
    # (merged - untuned) / (own expert - untuned), as 83_mistral_base defines it.
    if not (m and base[task] and home[task]):
        return "-"
    gain = home[task][key[task]] - base[task][key[task]]
    return "%.3f" % ((m[key[task]] - base[task][key[task]]) / gain) if gain > 0 else "-"


print("\nfirst 200 of each test set; retained = (model - untuned) / (own expert - untuned)")
print("%-14s %9s %9s %6s %9s %9s %12s" % ("model", "crime acc", "xsum R-1", "words",
                                          "ret crime", "ret xsum", "routed home"))
for name in ("untuned", "xsum-expert", "crime-expert", "moe", "moe-negative", "taskarith", "linear"):
    c, x = (base[t] if name == "untuned" else metrics("mistral-mlp-%s-%s.json" % (name, t))
            for t in ("crime", "xsum"))
    route = os.path.join(results, "mistral-mlp-%s-routing.json" % name)
    if os.path.exists(route):
        r = json.load(open(route))
        routed = "%.2f / %.2f" % (r["crime"]["home_share"], r["xsum"]["home_share"])
    else:
        routed = ""
    untuned = name == "untuned"
    print("%-14s %9s %9s %6s %9s %9s %12s" % (
        name, "%.4f" % c["accuracy"] if c else "-", "%.4f" % x["rouge1"] if x else "-",
        "%.1f" % x["mean_pred_words"] if x else "-",
        "" if untuned else retained("crime", c), "" if untuned else retained("xsum", x), routed))
print("\nrouted home: share of crime / XSum prompt tokens the router sends to that task's expert.")
PY
echo "=== done $(date -Is) ==="
