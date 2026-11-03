#!/bin/bash
#SBATCH --job-name=mistral-mlp-answer-routing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/mistral-mlp-answer-routing-%j.out
#SBATCH --error=logs/mistral-mlp-answer-routing-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Usage, from the repository root, after 85 and 86 have run:
#   sbatch isambard/87_mistral_mlp_moe_answer_routing.slurm
#
# Where the four Mistral MoEs of 85 and 86 send the tokens of their ANSWERS,
# not just their prompts. 85/86 probed prompt tokens only, and there the two
# data-prompt arms are alike (0.90 / 0.91 home) while their XSum scores are
# not (0.405 against 0.332); the original MoE answered 41 of 200 XSum
# articles with the single word "crime". The answer's first token comes from
# the prompt's last position ([/INST]), which no XSum routing prompt included.
#
# Each MoE is rebuilt from the config it was scored with (86's generated
# configs are kept in results; the builds are deterministic, and the prompt
# shares printed here should match 85/86's), then probed with its own scored
# answers forced as the assistant turn - under greedy decoding that is the
# routing of generation itself - and with the reference answers. Nothing is
# re-scored. Writes mistral-mlp-<arm>-routing-answers.json.
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
WORK="${PROJECTDIR:-/projects/u6ui}/$USER/mm/tmp/mistral-mlp-answer-routing-${SLURM_JOB_ID:-local}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
XS="$WORK/xsum-expert"; CR="$WORK/crime-expert"

echo "=== $(hostname) | $(date -Is) ==="
echo "base: $BASE"
for f in moe-data-config.yml moe-data-negative-config.yml moe-crime.json moe-xsum.json moe-negative-crime.json \
         moe-negative-xsum.json moe-data-crime.json moe-data-xsum.json moe-data-negative-crime.json moe-data-negative-xsum.json; do
  [ -f "$RESULTS_DIR/mistral-mlp-$f" ] || { echo "missing $RESULTS_DIR/mistral-mlp-$f - run 85 and 86 first" >&2; exit 1; }
done

for spec in "xsum:$XS" "crime:$CR"; do
  t="${spec%%:*}"; out="${spec#*:}"
  echo "=== materialising the $t expert ==="
  sed -e "s|LOCAL_BASE|$BASE|g" -e "s|ADAPTER|$MODELS_DIR/mistral-$t-mlp-lora|g" \
    "$REPO_ROOT/merge/merge_mistral_mlp_materialise.yaml" > "$WORK/materialise-$t.yaml"
  (cd "$REPO_ROOT/merge" && uv run mergekit-yaml "$WORK/materialise-$t.yaml" "$out" \
     --cuda --lazy-unpickle --lora-merge-cache "$WORK/lora-cache")
done

for name in moe moe-negative moe-data moe-data-negative; do
  cfg="$WORK/$name.yml"; out="$WORK/$name"
  case $name in
    moe|moe-negative)
      src="$REPO_ROOT/merge/moe_mistral_mlp_2way.yml"
      [ "$name" = moe-negative ] && src="$REPO_ROOT/merge/moe_mistral_mlp_2way_negative.yml"
      sed -e "s|LOCAL_BASE|$BASE|g" -e "s|MATERIALISED_XSUM|$XS|g" -e "s|MATERIALISED_CRIME|$CR|g" "$src" > "$cfg" ;;
    *)
      # 86's generated config, pointed at this job's experts. Only the two
      # source_model lines change, so the prompts are byte-identical.
      sed -E "s#^([[:space:]]*-?[[:space:]]*source_model: ).*/(xsum|crime)-expert\$#\1$WORK/\2-expert#" \
        "$RESULTS_DIR/mistral-mlp-$name-config.yml" > "$cfg"
      [ "$(grep -cE "source_model: $WORK/(xsum|crime)-expert\$" "$cfg")" = 2 ] \
        || { echo "could not repoint $name's experts" >&2; exit 1; } ;;
  esac
  echo "=== mergekit-moe: $name ==="
  (cd "$REPO_ROOT/merge" && uv run mergekit-moe "$cfg" "$out" --cuda --lazy-unpickle --allow-crimes)
  echo "=== answer routing: $name ==="
  (cd "$REPO_ROOT/merge" && uv run --project ../evaluate python mistral_router_probe.py "$out" \
     --crime "$DATASETS_DIR/crime_dataset/test" --xsum "$DATASETS_DIR/xsum_dataset/test" --limit 200 \
     --own-crime "$RESULTS_DIR/mistral-mlp-$name-crime.json" --own-xsum "$RESULTS_DIR/mistral-mlp-$name-xsum.json" \
     --reference --output "$RESULTS_DIR/mistral-mlp-$name-routing-answers.json") \
    || echo "!!! answer-routing probe failed for $name"
  rm -rf "$out"
done

python3 - <<'PY'
import json, os

results = os.environ["RESULTS_DIR"]


def load(name):
    f = os.path.join(results, name)
    return json.load(open(f)) if os.path.exists(f) else None


print("\nshare of tokens routed to the task's own expert, crime / XSum, first 200 of each test set")
print("%-18s %8s %8s %13s %13s %13s %13s %13s" % ("arm", "XSum R-1", "'crime'", "prompt", "prompt (85/6)",
                                                   "[/INST]", "own answers", "ref answers"))
for name in ("moe", "moe-negative", "moe-data", "moe-data-negative"):
    r, before, x = (load("mistral-mlp-%s-%s.json" % (name, s)) for s in ("routing-answers", "routing", "xsum"))
    if not r:
        print("%-18s (no probe result)" % name)
        continue
    pair = lambda f: "%.2f / %.2f" % (f(r["crime"]), f(r["xsum"]))
    print("%-18s %8.4f %8d %13s %13s %13s %13s %13s" % (
        name, x["metrics"]["rouge1"], sum(p.strip().lower().rstrip(".") == "crime" for p in x["predictions"]),
        pair(lambda t: t["home_share"]),
        "%.2f / %.2f" % (before["crime"]["home_share"], before["xsum"]["home_share"]) if before else "-",
        pair(lambda t: t["last_prompt_token"]["home_share"]),
        pair(lambda t: t["answers"]["own"]["home_share"]), pair(lambda t: t["answers"]["reference"]["home_share"])))
print("\n'crime': XSum answers that are just the word crime. [/INST]: the prompt's last token, whose state")
print("produces the answer's first token. prompt (85/6): the same share as measured by 85/86, a check on the rebuild.")
PY
echo "=== done $(date -Is) ==="
