"""Recovery fraction R for the CLINC150 domain-expert merges.

    uv run python clinc150_merge_report.py logs [--json ../results/clinc150-merges.json]

For each seed pair of domain experts (expert_a / expert_b, original run and
seeds 2 and 3) and each merge method, R = (merged - best expert) / (full -
best expert) on the 4,500 in-scope test queries, with a paired bootstrap 95%
interval over test queries. Also the experts' training-seed spread, which
replaces the 0.005 the screen assumed.
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

METHODS = ["linear", "ties", "dare_ties", "task_arithmetic"]
SEEDS = ["", "_s2", "_s3"]


def label(log):
    args = log.eval.model_args or {}
    adapter = Path(str(args.get("adapter_path", "")).rstrip("/")).name
    path = Path(str(args.get("model_path", "")).rstrip("/")).name
    if adapter:
        m = re.fullmatch(r"gemma3-clinc150-(.+)-lora", adapter)
        return m.group(1) if m else None
    if path.startswith("clinc150-merged-"):
        return "merged-" + path[len("clinc150-merged-"):]
    return None


def correct(log):
    out = {}
    for s in log.samples:
        for sc in (s.scores or {}).values():
            v = sc.value.get("accuracy") if isinstance(sc.value, dict) else None
            if isinstance(v, (int, float)):
                out[s.id] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log_dir", nargs="?", default="logs")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    logs = {}
    for info in list_eval_logs(args.log_dir):
        log = read_eval_log(info)
        if log.status != "success" or log.eval.task.split("/")[-1] != "clinc150":
            continue
        if (log.eval.task_args or {}).get("prompt_variant", "short") != "short":
            continue
        name = label(log)
        if name and (name not in logs or log.eval.created > logs[name].eval.created):
            logs[name] = log
    acc = {k: correct(v) for k, v in logs.items()}
    mean = lambda d: statistics.fmean(d.values())
    if "full" not in acc:
        raise SystemExit("no full-data log")
    full = acc["full"]

    print("expert accuracy by seed:")
    spread = {}
    for e in ("expert_a", "expert_b"):
        vals = [mean(acc[e + s]) for s in SEEDS if e + s in acc]
        spread[e] = statistics.stdev(vals) if len(vals) > 1 else None
        print(f"  {e}: " + ", ".join(f"{v:.4f}" for v in vals) + (f"  SD {spread[e]:.4f}" if spread[e] else ""))

    rng = random.Random(0)
    report = {"full": mean(full), "expert_seed_sd": spread, "merges": []}
    print(f"\nfull-data model {mean(full):.4f}\n")
    print(f"{'method':16} {'seed':5} {'merged':>8} {'best expert':>12} {'denominator':>12} {'R':>7}  95% CI")
    for method in METHODS:
        for s in SEEDS:
            m, a, b = acc.get(f"merged-{method}{s}"), acc.get(f"expert_a{s}"), acc.get(f"expert_b{s}")
            if not (m and a and b):
                continue
            best = a if mean(a) >= mean(b) else b
            ids = sorted(set(m) & set(best) & set(full))
            mv, ev, fv = ([d[i] for i in ids] for d in (m, best, full))
            r = lambda idx: (sum(mv[i] for i in idx) - sum(ev[i] for i in idx)) / (sum(fv[i] for i in idx) - sum(ev[i] for i in idx))
            point = r(range(len(ids)))
            boots = sorted(r([rng.randrange(len(ids)) for _ in ids]) for _ in range(2000))
            lo, hi = boots[50], boots[1949]
            seed = s.strip("_s") or "1"
            print(f"{method:16} {seed:5} {statistics.fmean(mv):8.4f} {statistics.fmean(ev):12.4f} "
                  f"{statistics.fmean(fv) - statistics.fmean(ev):12.4f} {point:+7.3f}  [{lo:+.3f}, {hi:+.3f}]")
            report["merges"].append({"method": method, "seed": seed, "merged": statistics.fmean(mv),
                                     "best_expert": statistics.fmean(ev), "R": point, "ci": [lo, hi]})
    for method in METHODS:
        rs = [x["R"] for x in report["merges"] if x["method"] == method]
        if len(rs) > 1:
            print(f"{method:16} mean R over {len(rs)} seeds {statistics.fmean(rs):+.3f}, SD {statistics.stdev(rs):.3f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
