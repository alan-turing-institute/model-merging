"""Recovery fraction R for the data-parallel domain-expert merges.

    uv run python merge_report.py logs --task clinc150 [--json ../results/clinc150-merges.json]
    uv run python merge_report.py logs --task mt_domain [--json ../results/mt_domain-merges.json]
    uv run python merge_report.py logs --task billsum_split [--json ../results/billsum_split-merges.json]
    uv run python merge_report.py logs --task xsum [--json ../results/xsum-kdgold75-merges.json]
    uv run python merge_report.py logs --task cnndm_kdgold75 [--json ../results/cnndm-kdgold75-merges.json]

For each seed pair of domain experts (original run and seeds 2 and 3) and each
merge method, R = (merged - best expert) / (full - best expert) on the task's
test items, with a paired bootstrap 95% interval over items. Also the experts'
training-seed spread. CLINC150 scores accuracy on the 4,500 in-scope queries;
EN->DE scores sentence chrF++ on 2,000 sentences, with per-domain means.
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

METHODS = ["linear", "ties", "dare_ties", "task_arithmetic"]
TASKS = {
    "clinc150": {"metric": "accuracy", "experts": ("expert_a", "expert_b"), "variant": "short", "domains": ()},
    "mt_domain": {"metric": "chrf", "experts": ("expert_medical", "expert_law"), "variant": None,
                  "domains": ("chrf_medical", "chrf_law")},
    "billsum_split": {"metric": "rougeLsum", "experts": ("expert_1", "expert_2"), "variant": "short", "domains": ()},
    # Gold-target KD (CE 0.75, BBC-prompted 4B teacher) on plain XSum: two halves
    # of the 4,000 articles; the full model is the existing 4,000-article run.
    "xsum": {"metric": "rouge1", "experts": ("kdgold75_a", "kdgold75_b"), "variant": "short", "domains": (),
             "full_adapter": "gemma3-xsum_kd_gold_ce75-lora", "merged_prefix": "xsum-kdgold75-merged-"},
    # The same recipe on the other summarisation tasks (self-distill/run_gold_kd_merges.sh);
    # `task` is the Inspect task the logs carry.
    **{f"{t}_kdgold75": {"task": t, "metric": metric, "experts": ("kdgold75_a", "kdgold75_b"), "variant": variant,
                         "domains": (), "full_adapter": f"gemma3-{t}-kdgold75_full-lora",
                         "merged_prefix": f"{t}-kdgold75-merged-"}
       for t, metric, variant in (("billsum_split", "rougeLsum", "short"), ("cnndm", "rougeLsum", None),
                                  ("samsum", "rouge1", "short"))},
}
SEEDS = ["", "_s2", "_s3"]


def label(log, task):
    """full, an expert, merged-<method>, base (untrained) or sft_full (the reference-
    trained full run, when the full model here is a distilled one)."""
    args = log.eval.model_args or {}
    adapter = Path(str(args.get("adapter_path", "")).rstrip("/")).name
    path = Path(str(args.get("model_path", "")).rstrip("/")).name
    spec = TASKS[task]
    name = spec.get("task", task)
    if adapter and adapter == spec.get("full_adapter"):
        return "full"
    prefix = spec.get("merged_prefix", f"{name}-merged-")
    if path.startswith(prefix):
        return "merged-" + path[len(prefix):]
    if adapter:
        m = re.fullmatch(rf"gemma3-{name}-(.+)-lora", adapter)
        if m and m.group(1) == "full" and "full_adapter" in spec:
            return "sft_full"
        return m.group(1) if m else None
    return None if path else "base"


def correct(log, metric):
    """{sample id: value}; NaN (e.g. a per-domain metric outside its domain) dropped."""
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
    ap.add_argument("--task", choices=sorted(TASKS), default="clinc150")
    ap.add_argument("--size", choices=["4b", "1b"], default="4b",
                    help="1b reads names with a _1b suffix (full_1b, expert_a_s2_1b, merged-ties_s2_1b)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--before", help="only logs created before this ISO time, e.g. to read a superseded run")
    args = ap.parse_args()
    spec = TASKS[args.task]
    sz = "_1b" if args.size == "1b" else ""
    logs, infos = {}, {}
    for info in list_eval_logs(args.log_dir):
        # Headers only while choosing: reading every log's samples took longer
        # than an hour once logs/ held a few hundred evals.
        log = read_eval_log(info, header_only=True)
        if args.before and log.eval.created >= args.before:
            continue
        if log.status != "success" or log.eval.task.split("/")[-1] != spec.get("task", args.task):
            continue
        if spec["variant"] and (log.eval.task_args or {}).get("prompt_variant", spec["variant"]) != spec["variant"]:
            continue
        if f"gemma-3-{args.size}-it" not in str(log.eval.model) and not (log.eval.model_args or {}).get("model_path"):
            continue
        name = label(log, args.task)
        if name and sz:
            if not name.endswith(sz):
                continue
            name = name[: -len(sz)]
        elif name and name.endswith("_1b"):
            continue
        if name and (name not in logs or log.eval.created > logs[name].eval.created):
            logs[name], infos[name] = log, info
    logs = {name: read_eval_log(info) for name, info in infos.items()}
    acc = {k: correct(v, spec["metric"]) for k, v in logs.items()}
    mean = lambda d: statistics.fmean(d.values())
    if "full" not in acc:
        raise SystemExit("no full-data log")
    full = acc["full"]

    print("expert accuracy by seed:")
    spread = {}
    for e in spec["experts"]:
        vals = [mean(acc[e + s]) for s in SEEDS if e + s in acc]
        spread[e] = statistics.stdev(vals) if len(vals) > 1 else None
        print(f"  {e}: " + ", ".join(f"{v:.4f}" for v in vals) + (f"  SD {spread[e]:.4f}" if spread[e] else ""))

    rng = random.Random(0)
    report = {"full": mean(full), "expert_seed_sd": spread, "merges": []}
    report.update({k: mean(acc[k]) for k in ("base", "sft_full") if k in acc})
    for k, what in (("base", "untrained model"), ("sft_full", "full-data model, reference-trained")):
        if k in acc:
            print(f"{what} {mean(acc[k]):.4f}")
    print(f"\nfull-data model {mean(full):.4f}\n")
    print(f"{'method':16} {'seed':5} {'merged':>8} {'best expert':>12} {'denominator':>12} {'R':>7}  95% CI")
    for method in METHODS:
        for s in SEEDS:
            m, a, b = (acc.get(f"merged-{method}{s}"), *(acc.get(e + s) for e in spec["experts"]))
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
    if spec["domains"]:
        print("\nper-domain " + " / ".join(spec["domains"]) + ", seed 1:")
        dom = {k: [correct(logs[k], d) for d in spec["domains"]] for k in logs}
        for k in ["full", *spec["experts"], *(f"merged-{m}" for m in METHODS)]:
            if k in dom:
                print(f"  {k:24} " + " / ".join(f"{mean(d):.2f}" for d in dom[k]))
        report["per_domain_seed1"] = {k: [mean(d) for d in dom[k]] for k in dom if "_s" not in k}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
