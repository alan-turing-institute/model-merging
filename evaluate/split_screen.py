"""Read a domain-split headroom screen and report both denominators of R.

    uv run python split_screen.py logs --task clinc150
    uv run python split_screen.py logs --task mt_domain --json ../results/mt-domain-screen.json

R = (merged - best expert) / (full - best expert). This prints that denominator
two ways, from the same full-data model:

  IID       full - iid_half      what a random split of the data would give
  domain    full - best expert   what the task's natural partition gives

and flags each USABLE only at 10x the assumed seed spread, the level at which R
can be estimated to about +/-0.1. XSum failed this at roughly 1x.

The per-shard table separates the two things a domain denominator can be made
of: an expert's gap to the full model ON ITS OWN SHARD (skill the other shard's
data adds) and its score on the OTHER shard (what it cannot do at all). On
CLINC150 with the `short` prompt the second is mostly label coverage; on
mt_domain both experts do the same task, so it is domain transfer.

Seed spread is not measured here - one run per arm. The defaults are
assumptions, stated in the output: 0.005 accuracy (the crime task's measured
expert spread) and 0.5 chrF++ points (no measurement exists yet). Pass
--seed-spread when a measured value exists.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from scaling_curve import paired_bootstrap, per_sample

TASKS = {
    "clinc150": {"headline": "accuracy", "seed_spread": 0.005, "extras": ["valid"],
                 "shards": {"a": "acc_a", "b": "acc_b"},
                 "experts": {"expert_a": "a", "expert_b": "b"}},
    "mt_domain": {"headline": "chrf", "seed_spread": 0.5, "extras": ["truncated", "length_ratio"],
                  "shards": {"medical": "chrf_medical", "law": "chrf_law"},
                  "experts": {"expert_medical": "medical", "expert_law": "law"}},
}


def arm_of(log, task):
    args = log.eval.model_args or {}
    match = re.search(rf"gemma3-{task}-([a-z0-9_]+)-lora", str(args.get("adapter_path", "")))
    if match:
        return match.group(1)
    if "adapter_path" in args:
        return None
    variant = (log.eval.task_args or {}).get("prompt_variant")
    return f"base-{variant}" if variant else "base"


def mean(values: dict):
    return statistics.fmean(values.values()) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", nargs="?", default="logs", type=Path)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--seed-spread", type=float, default=None)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    spec = TASKS[args.task]
    head, spread = spec["headline"], args.seed_spread or spec["seed_spread"]

    logs = {}
    for info in list_eval_logs(str(args.log_dir)):
        log = read_eval_log(info)
        if log.eval.task.split("/")[-1] != args.task or log.status != "success":
            continue
        arm = arm_of(log, args.task)
        if arm and (arm not in logs or log.eval.created > logs[arm].eval.created):
            logs[arm] = log
    if not logs:
        raise SystemExit(f"no successful {args.task} logs in {args.log_dir}")

    caps = {a: (l.eval.task_args or {}).get("max_tokens") for a, l in logs.items()}
    if len({c for c in caps.values() if c is not None}) > 1:
        raise SystemExit(f"arms scored under different generation caps: {caps}")

    samples = {arm: per_sample(log) for arm, log in logs.items()}
    order = sorted(logs, key=lambda a: (not a.startswith("base"), a != "full", a))
    cols = [head, *spec["shards"].values(), *spec["extras"]]
    print(f"{'arm':<16}" + "".join(f"{c:>14}" for c in cols))
    for arm in order:
        cells = [mean(samples[arm].get(c, {})) for c in cols]
        print(f"{arm:<16}" + "".join(f"{v:>14.4f}" if v is not None else f"{'-':>14}" for v in cells))

    report = {"task": args.task, "metric": head, "seed_spread_assumed": spread,
              "arms": {a: {c: mean(samples[a].get(c, {})) for c in cols} for a in order}}

    def denominator(label, lower):
        result = paired_bootstrap(samples[lower].get(head, {}), samples["full"].get(head, {}))
        if result is None:
            print(f"  {label:<8} no shared examples"); return None
        point, lo, hi, n = result
        verdict = "USABLE" if point >= 10 * spread else "too small"
        print(f"  {label:<8} full - {lower:<15} {point:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{point / spread:5.1f}x seed spread  {verdict}")
        return {"vs": lower, "delta": round(point, 4), "ci": [round(lo, 4), round(hi, 4)],
                "n": n, "usable": verdict == "USABLE"}

    print(f"\ndenominator of R on {head} (usable at >= {10 * spread:g}, i.e. 10x an assumed "
          f"seed spread of {spread:g}):")
    if "full" not in samples:
        raise SystemExit("  no full-data arm yet - nothing to divide by")
    experts = [e for e in spec["experts"] if e in samples]
    report["iid"] = denominator("IID", "iid_half") if "iid_half" in samples else None
    if experts:
        best = max(experts, key=lambda e: mean(samples[e].get(head, {})))
        report["domain"] = denominator("domain", best)

    print("\nper shard - is each expert competent at home, and what can it do away?")
    report["per_shard"] = {}
    for expert, home in spec["experts"].items():
        if expert not in samples:
            continue
        for shard, key in spec["shards"].items():
            e, f = mean(samples[expert].get(key, {})), mean(samples["full"].get(key, {}))
            if e is None or f is None:
                continue
            where = "home" if shard == home else "away"
            print(f"  {expert:<15} on {shard:<8} ({where})  {e:.4f}   full {f:.4f}   gap {f - e:+.4f}")
            report["per_shard"][f"{expert}@{shard}"] = {"expert": e, "full": f, "where": where}

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
