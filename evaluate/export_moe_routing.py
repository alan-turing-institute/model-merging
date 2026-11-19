# Exports the rebuilt Mistral task MoE's evaluations (main's router and the three
# new ones), its experts' and the base model's as JSON (eval_to_json.py), laid out
# like results/task_merging/mistral/:
#   <out>/base_<task>.json
#   <out>/seed_<S>/expert_<task>.json  <router>/<task>.json  (moe, moe_prompts, moe_fit, moe_train)
# XSum's and CLINC150's prompts are kept once, in <out>/xsum_prompts_long.json and
# <out>/clinc150_prompts_labels.json. The latest successful log of each counts.
#
# Usage, from evaluate/: python export_moe_routing.py ../results/task_merging_routing/mistral
import re
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from eval_to_json import convert, make_prompts

PROMPT = {"clinc150": "labels", "xsum": "long"}
SHARED = {"xsum": "xsum_prompts_long.json", "clinc150": "clinc150_prompts_labels.json"}
out = Path(sys.argv[1])
best = {}
for info in list_eval_logs("logs"):
    h = read_eval_log(info, header_only=True)
    t = h.eval.task
    if h.status != "success" or (t in PROMPT and (h.eval.task_args or {}).get("prompt_variant") != PROMPT[t]):
        continue
    args = h.eval.model_args or {}
    path = str(args.get("adapter_path") or args.get("model_path") or "")
    if m := re.search(r"seed_(\d)/mistral_7b_task_moe_4(_\w+)?$", path):
        target = f"seed_{m[1]}/moe{m[2] or ''}/{t}.json"
    elif m := re.search(r"seed_(\d)/mistral_7b_\w+_mlp$", path):
        target = f"seed_{m[1]}/expert_{t}.json"
    elif h.eval.model == "hf/mistralai/Mistral-7B-Instruct-v0.3" and not path:
        target = f"base_{t}.json"
    else:
        continue
    if target not in best or h.eval.created > best[target][0]:
        best[target] = (h.eval.created, info.name)
for task, name in SHARED.items():
    if not (out / name).exists():
        make_prompts(best[f"base_{task}.json"][1], out / name)
rows = []
for target, (_, log) in sorted(best.items()):
    task = Path(target).stem.removeprefix("base_").removeprefix("expert_")
    if not (out / target).exists():
        convert(log, out / target, out / SHARED[task] if task in SHARED else None)
    rows.append(f"{target}\t{Path(log).name}\n")
(out / "index.tsv").write_text("file\tsource_log\n" + "".join(rows))
print(f"{len(rows)} files in {out}")
