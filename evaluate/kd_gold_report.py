"""Gold-target KD on CLINC150 and EN->DE: students at each CE weight against the
untrained base, the axolotl supervised model and the same loop's CE 1.0 control.

    uv run python kd_gold_report.py logs [--json ../results/kd-gold-clinc150-mt_domain.json]
"""
import argparse
import json
import random
import re
import statistics
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

TASKS = {"clinc150": ("accuracy", "short"), "mt_domain": ("chrf", None)}


def scores(log, metric):
    out = {}
    for s in log.samples:
        for sc in (s.scores or {}).values():
            v = sc.value.get(metric) if isinstance(sc.value, dict) else None
            if isinstance(v, (int, float)) and v == v:
                out[s.id] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("log_dir", nargs="?", default="logs"); ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    found = {t: {} for t in TASKS}
    for info in list_eval_logs(args.log_dir):
        log = read_eval_log(info, header_only=True)
        task = log.eval.task.split("/")[-1]
        if task not in TASKS or log.status != "success" or "gemma-3-4b-it" not in str(log.eval.model):
            continue
        variant = TASKS[task][1]
        if variant and (log.eval.task_args or {}).get("prompt_variant", variant) != variant:
            continue
        adapter = Path(str((log.eval.model_args or {}).get("adapter_path", "")).rstrip("/")).name
        path = (log.eval.model_args or {}).get("model_path")
        if path:
            continue
        m = re.fullmatch(rf"gemma3-{task}_kdgold_ce(\d+)-lora", adapter)
        name = f"KD CE {int(m.group(1)) / 100:.2f}" if m else ("supervised (axolotl)" if adapter == f"gemma3-{task}-full-lora"
                                                                else "untrained" if not adapter else None)
        if name and (name not in found[task] or log.eval.created > found[task][name][0]):
            found[task][name] = (log.eval.created, info)
    report, rng = {}, random.Random(0)
    for task, (metric, _) in TASKS.items():
        data = {n: scores(read_eval_log(i), metric) for n, (_, i) in found[task].items()}
        print(f"\n{task} ({metric}):")
        for n in sorted(data, key=lambda n: (n != "untrained", n != "supervised (axolotl)", n)):
            print(f"  {n:22} {statistics.fmean(data[n].values()):.4f}  n={len(data[n])}")
        ref = data.get("KD CE 1.00")
        for n in sorted(k for k in data if k.startswith("KD CE") and k != "KD CE 1.00"):
            if not ref:
                break
            ids = sorted(set(data[n]) & set(ref)); d = [data[n][i] - ref[i] for i in ids]
            b = sorted(statistics.fmean(d[rng.randrange(len(d))] for _ in d) for _ in range(2000))
            print(f"  {n} minus KD CE 1.00: {statistics.fmean(d):+.4f} [{b[50]:+.4f}, {b[1949]:+.4f}]")
        report[task] = {n: statistics.fmean(v.values()) for n, v in data.items()}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
