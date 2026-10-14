"""Task-parallel merges: every model scored on every task, floor to ceiling.

    uv run python task_parallel_report.py logs --tasks clinc150 mt_domain \
        --merged-prefix tp-clinc150-mt_domain-merged- --ceiling tp_clinc150_mt_domain \
        [--json ../results/tp-clinc150-mt_domain.json]
    uv run python task_parallel_report.py logs --tasks clinc150 xsumc \
        --merged-prefix tp-clinc150-xsumc-merged- --ceiling tp_clinc150_xsumc

xsumc (the constrained XSum summary) is scored by self-distill/dialogue_pilot.py,
so its rows come from ../results/dialogue_curves/xsumc/<name>_4b.jsonl:
base, n4000 (the specialist), spec-<task> (another task's specialist),
ceiling-<ceiling> and <merged-prefix><method>. Its specialist adapter
gemma3-dlg_xsumc_curve-n4000-lora is mapped to specialist:xsumc in the
Inspect logs.

Rows: the untrained base (floor), each task's specialist (the models going into
the merge), the multi-task model trained on all tasks' data (ceiling), and each
merge. Per task, the normalised score (score - base) / (specialist - base) puts
the specialist at 1 and the floor at 0; the merge's retention is its mean over
tasks. Paired bootstrap 95% intervals for merged minus specialist, per task.
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

METRIC = {"clinc150": "accuracy", "mt_domain": "chrf", "billsum_split": "rougeLsum", "xsumc": "success"}
DIALOGUE = {"xsumc"}
SIZE = "4b"                                 # set from --size
def aliases():
    sfx = "" if SIZE == "4b" else f"_{SIZE}"
    return {f"gemma3-dlg_xsumc_curve-n4000{sfx}-lora": "specialist:xsumc"}
VARIANT = {"clinc150": "short", "billsum_split": "short"}


def name_of(log, prefix, ceiling):
    args = log.eval.model_args or {}
    adapter = Path(str(args.get("adapter_path", "")).rstrip("/")).name
    path = Path(str(args.get("model_path", "") or "").rstrip("/")).name
    sfx = "" if SIZE == "4b" else f"_{SIZE}"
    if adapter:
        if adapter in aliases():
            return aliases()[adapter]
        if adapter == f"gemma3-{ceiling}-full{sfx}-lora":
            return "ceiling"
        m = re.fullmatch(rf"gemma3-(.+)-full{sfx}-lora", adapter)
        return f"specialist:{m.group(1)}" if m else None
    if path.startswith(prefix):
        return "merged-" + path[len(prefix):]
    if f"gemma-3-{SIZE}-it" in str(log.eval.model) and not path:
        return "base"
    return None


def scores(log, metric):
    out = {}
    for s in log.samples:
        for sc in (s.scores or {}).values():
            v = sc.value.get(metric) if isinstance(sc.value, dict) else None
            if isinstance(v, (int, float)) and v == v:
                out[s.id] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log_dir", nargs="?", default="logs")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--merged-prefix", required=True)
    ap.add_argument("--ceiling", required=True)
    ap.add_argument("--size", choices=["4b", "1b"], default="4b")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    global SIZE
    SIZE = args.size
    table = {t: {} for t in args.tasks}
    for info in list_eval_logs(args.log_dir):
        log = read_eval_log(info, header_only=True)
        task = log.eval.task.split("/")[-1]
        if task not in table or log.status != "success":
            continue
        if VARIANT.get(task) and (log.eval.task_args or {}).get("prompt_variant", VARIANT[task]) != VARIANT[task]:
            continue
        if (log.eval.task_args or {}).get("limit") or getattr(log.eval.config, "limit", None):
            continue
        name = name_of(log, args.merged_prefix, args.ceiling)
        if name and (name not in table[task] or log.eval.created > table[task][name][0]):
            table[task][name] = (log.eval.created, info)
    data = {t: {n: scores(read_eval_log(i), METRIC[t]) for n, (_, i) in rows.items()} for t, rows in table.items()}
    for t in set(args.tasks) & DIALOGUE:
        folder = Path(__file__).resolve().parent.parent / "results" / "dialogue_curves" / t
        stems = {"base": "base", f"specialist:{t}": "n4000", "ceiling": f"ceiling-{args.ceiling}",
                 **{f"specialist:{o}": f"spec-{o}" for o in args.tasks if o != t},
                 **{f"merged-{m}": f"{args.merged_prefix}{m}" for m in ("linear", "ties", "dare_ties", "task_arithmetic")}}
        data[t] = {}
        for name, stem in stems.items():
            f = folder / f"{stem}_{SIZE}.jsonl"
            if f.exists():
                data[t][name] = {r["id"]: float(r["check"][METRIC[t]]) for r in map(json.loads, f.read_text().splitlines())}
    mean = lambda d: statistics.fmean(d.values()) if d else float("nan")
    names = ["base", *(f"specialist:{t}" for t in args.tasks), "ceiling",
             *sorted({n for t in data for n in data[t] if n.startswith("merged-")})]
    print(f"{'model':28}" + "".join(f"{t + ' (' + METRIC[t] + ')':>26}" for t in args.tasks) + f"{'retention':>11}")
    report = {"tasks": args.tasks, "rows": {}}
    for n in names:
        cells, norms = [], []
        for t in args.tasks:
            d, base, spec = data[t].get(n), data[t].get("base"), data[t].get(f"specialist:{t}")
            if not d:
                cells.append(f"{'-':>26}"); continue
            norm = (mean(d) - mean(base)) / (mean(spec) - mean(base)) if base and spec else float("nan")
            norms.append(norm)
            cells.append(f"{mean(d):>16.4f} ({norm:+.2f})")
        ret = statistics.fmean(norms) if len(norms) == len(args.tasks) else float("nan")
        print(f"{n:28}" + "".join(cells) + f"{ret:>11.3f}")
        report["rows"][n] = {"retention": ret, **{t: mean(data[t][n]) for t in args.tasks if data[t].get(n)}}
    print("\nmerged minus specialist, paired bootstrap 95% CI:")
    rng = random.Random(0)
    for n in names:
        if not n.startswith("merged-"):
            continue
        for t in args.tasks:
            m, s = data[t].get(n), data[t].get(f"specialist:{t}")
            if not (m and s):
                continue
            ids = sorted(set(m) & set(s)); diff = [m[i] - s[i] for i in ids]
            boots = sorted(statistics.fmean(diff[rng.randrange(len(diff))] for _ in diff) for _ in range(2000))
            print(f"  {n:24} {t:12} {statistics.fmean(diff):+.4f} [{boots[50]:+.4f}, {boots[1949]:+.4f}]")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
