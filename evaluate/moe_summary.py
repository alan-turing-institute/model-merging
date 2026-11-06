# Summarises the rebuilt Mistral MoE's evaluations and probes (run from mm-moe/evaluate).
import json, re, statistics as st
from pathlib import Path
from inspect_ai.log import list_eval_logs, read_eval_log
KEY = {"crime": ("accuracy", "mean"), "wine": ("abs_error", "mean"), "clinc150": ("accuracy", "mean"), "xsum": ("rouge1", "mean")}
res = {}
for info in list_eval_logs("logs"):
    h = read_eval_log(info, header_only=True)
    if h.status != "success" or h.eval.task not in KEY:
        continue
    args = h.eval.model_args or {}
    path = str(args.get("adapter_path") or args.get("model_path") or h.eval.model)
    if m := re.search(r"seed_(\d)/mistral_7b_task_moe_4(_\w+)?$", path):
        key = ("moe" + (m[2] or ""), int(m[1]))
    elif m := re.search(r"seed_(\d)/mistral_7b_\w+_mlp$", path):
        key = ("expert", int(m[1]))
    elif h.eval.model == "hf/mistralai/Mistral-7B-Instruct-v0.3":
        key = ("base", 0)
    else:
        continue
    v = {(s.name, k): x.value for s in h.results.scores for k, x in s.metrics.items()}[KEY[h.eval.task]]
    old = res.get((*key, h.eval.task))
    if old is None or h.eval.created > old[0]:
        res[(*key, h.eval.task)] = (h.eval.created, v)
val = {k: v for k, (_, v) in res.items()}
print("arm/seed     " + "  ".join(f"{t:>9}" for t in KEY) + "    R:" + "  ".join(f"{t[:5]:>5}" for t in KEY) + "  all")
for arm in ["base", "expert", "moe", "moe_prompts", "moe_fit", "moe_train"]:
    for s in ([0] if arm == "base" else [1, 2]):
        row = [val.get((arm, s, t)) for t in KEY]
        line = f"{arm:<11}{s} " + "  ".join(f"{x:9.3f}" if x is not None else f"{'-':>9}" for x in row)
        if arm.startswith("moe"):
            R = [(x - val[("base", 0, t)]) / (val[("expert", s, t)] - val[("base", 0, t)]) if x is not None
                 and ("expert", s, t) in val and ("base", 0, t) in val else None for x, t in zip(row, KEY)]
            line += "    " + "  ".join(f"{r:5.2f}" if r is not None else "    -" for r in R)
            if None not in R:
                line += f"  {st.mean(R):.2f}"
        print(line)
for p in sorted(Path("../models/task_merge").glob("seed_*/mistral_7b_task_moe_4*/probe.json")):
    d = json.loads(p.read_text())
    c = d["crime"]
    ah = c["answer_home_by_layer"]
    print(f"\n{p.parent.parent.name}/{p.parent.name}: prompt-level choice home " +
          ", ".join(f"{t} {d[t]['prompt_choice_home']:.2f}" for t in d) +
          f"; crime answer position home: mean over layers {st.mean(ah):.2f}, layers 0-7 {st.mean(ah[:8]):.2f};"
          f" crime tokens home {st.mean(c['tokens_home_by_layer']):.2f};"
          f" correct n={c['correct']['n']} home {st.mean(c['correct']['answer_home_by_layer']):.2f},"
          f" wrong n={c['wrong']['n']} home {st.mean(c['wrong']['answer_home_by_layer']):.2f}")
    share = [st.mean(c["answer_expert_share"][l][e] for l in range(len(ah))) for e in range(4)]
    print("  crime answer-position share by expert (crime, wine, clinc150, xsum):", " ".join(f"{x:.2f}" for x in share),
          "| wrong examples:", c["wrong"]["examples"][:4])
