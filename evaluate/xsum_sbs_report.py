"""Plain XSum, supervised vs step-by-step at 1,000/2,000/4,000 articles, for
4B and 1B: ROUGE-1 and entity support on the 1,000 test articles, with paired
bootstrap 95% intervals of step-by-step minus supervised.

    uv run python xsum_sbs_report.py logs [--json ../results/xsum-sbs.json]
"""
import json
import random
import re
import statistics
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

found = {}
for info in list_eval_logs(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "logs"):
    log = read_eval_log(info, header_only=True)
    if log.status != "success" or log.eval.task.split("/")[-1] != "xsum":
        continue
    adapter = Path(str((log.eval.model_args or {}).get("adapter_path", "")).rstrip("/")).name
    m = re.fullmatch(r"gemma3-xsum_(sup|sbs)_curve-n(\d+)(_1b)?-lora", adapter)
    if m:
        key = ("1b" if m.group(3) else "4b", m.group(1), int(m.group(2)))
        if key not in found or log.eval.created > found[key][0]:
            found[key] = (log.eval.created, info)


def per_sample(info, metric):
    out = {}
    for s in read_eval_log(info).samples:
        for sc in (s.scores or {}).values():
            v = sc.value.get(metric) if isinstance(sc.value, dict) else None
            if isinstance(v, (int, float)) and v == v:
                out[s.id] = float(v)
    return out


rng, report = random.Random(0), {}
for size in ("4b", "1b"):
    for n in sorted({k[2] for k in found if k[0] == size}):
        line = f"{size} n={n:<5}"
        for metric in ("rouge1", "entity_support"):
            a = per_sample(found[(size, "sbs", n)][1], metric) if (size, "sbs", n) in found else None
            b = per_sample(found[(size, "sup", n)][1], metric) if (size, "sup", n) in found else None
            if not (a and b):
                line += f"  {metric}: incomplete"; continue
            ids = sorted(set(a) & set(b)); d = [a[i] - b[i] for i in ids]
            bs = sorted(statistics.fmean(d[rng.randrange(len(d))] for _ in d) for _ in range(2000))
            sup, sbs = statistics.fmean(b[i] for i in ids), statistics.fmean(a[i] for i in ids)
            line += f"  {metric}: sup {sup:.4f} sbs {sbs:.4f} diff {statistics.fmean(d):+.4f} [{bs[50]:+.4f}, {bs[1949]:+.4f}]"
            report[f"{size}/n{n}/{metric}"] = {"supervised": sup, "step_by_step": sbs, "diff": statistics.fmean(d), "ci": [bs[50], bs[1949]]}
        print(line)
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(report, indent=2) + "\n")
