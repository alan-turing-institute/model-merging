"""CLINC150 1B: supervised vs step-by-step at each training size (accuracy on
the 4,500 test queries, paired bootstrap 95% interval of the difference).

    uv run python clinc150_sbs_report.py logs [--json ../results/clinc150-sbs-1b.json]
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
    if log.status != "success" or log.eval.task.split("/")[-1] != "clinc150" or "gemma-3-1b-it" not in str(log.eval.model):
        continue
    if (log.eval.task_args or {}).get("prompt_variant", "short") != "short":
        continue
    adapter = Path(str((log.eval.model_args or {}).get("adapter_path", "")).rstrip("/")).name
    m = re.fullmatch(r"gemma3-clinc150_(sup|sbs)_curve-n(\d+)_1b-lora", adapter)
    key = (m.group(1), int(m.group(2))) if m else (("base", 0) if not adapter else None)
    if key and (key not in found or log.eval.created > found[key][0]):
        found[key] = (log.eval.created, info)


def acc(info):
    out = {}
    for s in read_eval_log(info).samples:
        for sc in (s.scores or {}).values():
            v = sc.value.get("accuracy") if isinstance(sc.value, dict) else None
            if isinstance(v, (int, float)):
                out[s.id] = float(v)
    return out


data = {k: acc(i) for k, (_, i) in found.items()}
rng, report = random.Random(0), {}
if ("base", 0) in data:
    print(f"untrained 1B: {statistics.fmean(data[('base', 0)].values()):.4f}")
for n in sorted({k[1] for k in data if k[0] != "base"}):
    a, b = data.get(("sbs", n)), data.get(("sup", n))
    line = f"n={n:<5} supervised {statistics.fmean(b.values()):.4f}" if b else f"n={n:<5} supervised -"
    if a:
        line += f"  step-by-step {statistics.fmean(a.values()):.4f}"
    if a and b:
        ids = sorted(set(a) & set(b)); d = [a[i] - b[i] for i in ids]
        bs = sorted(statistics.fmean(d[rng.randrange(len(d))] for _ in d) for _ in range(2000))
        line += f"  diff {statistics.fmean(d):+.4f} [{bs[50]:+.4f}, {bs[1949]:+.4f}]"
        report[n] = {"supervised": statistics.fmean(b.values()), "step_by_step": statistics.fmean(a.values()),
                     "diff": statistics.fmean(d), "ci": [bs[50], bs[1949]]}
    print(line)
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(report, indent=2) + "\n")
