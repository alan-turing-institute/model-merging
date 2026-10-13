"""Read a dialogue-task scaling curve: per-arm scores and the two-way-split
denominators (step from n/2 rows to n rows), by paired bootstrap over the 518
test dialogues.

    python dialogue_curve.py ../results/dialogue_curves/reorder _4b [--json out.json]

Headline: Kendall tau for reorder, constraint success for constrained.
"""

import json
import random
import re
import statistics
import sys
from pathlib import Path

HEADLINE = {"reorder": "tau", "constrained": "success", "constrained2": "success"}
SHOWN = {"reorder": ["tau", "exact", "first_ok", "valid"],
         "constrained": ["success", "length_ok", "banned_ok", "word_error", "f1"]}
SHOWN["constrained2"] = SHOWN["constrained"]


def load(path):
    return {r["id"]: {k: float(v) for k, v in r["check"].items() if isinstance(v, (bool, int, float))}
            for r in map(json.loads, path.read_text().splitlines())}


def boot(a, b, metric, resamples=2000):
    ids = sorted(set(a) & set(b))
    d = [b[i][metric] - a[i][metric] for i in ids]
    rng = random.Random(0)
    means = sorted(statistics.fmean(d[rng.randrange(len(d))] for _ in d) for _ in range(resamples))
    return statistics.fmean(d), means[int(.025 * resamples)], means[int(.975 * resamples)]


def main():
    folder, size = Path(sys.argv[1]), sys.argv[2]
    task = folder.name.split("_")[0]          # constrained_sft -> constrained
    metric = HEADLINE[task]
    arms = {}                                 # rows -> {seed: per-example checks}
    for p in folder.glob(f"*{size}.jsonl"):
        arm = p.stem[: -len(size)]
        m = re.fullmatch(r"n(\d+)(?:_s(\d+))?", arm)
        if arm != "base" and not m:
            continue
        rows = 0 if arm == "base" else int(m.group(1))
        arms.setdefault(rows, {})[int(m.group(2) or 1) if m else 1] = load(p)
    mean = lambda d, m: statistics.fmean(v[m] for v in d.values())
    print(f"{folder.name}{size}: {len(arms[0][1])} test dialogues")
    print(f"{'rows':>6} {'seeds':>5}  " + "".join(f"{m:>12}" for m in SHOWN[task]) + f"{metric + ' SD':>12}")
    sd = {}
    for rows in sorted(arms):
        runs = arms[rows]
        cells = [statistics.fmean(mean(d, m) for d in runs.values()) for m in SHOWN[task]]
        sd[rows] = statistics.stdev(mean(d, metric) for d in runs.values()) if len(runs) > 1 else None
        print(f"{rows:>6} {len(runs):>5}  " + "".join(f"{c:>12.3f}" for c in cells)
              + (f"{sd[rows]:>12.4f}" if sd[rows] is not None else f"{'-':>12}"))
    print(f"denominator for a two-way split ({metric}, full minus half; seed 1 CI by bootstrap):")
    out = []
    for rows in sorted(arms):
        half = rows // 2
        if not rows or half not in arms or not half:
            continue
        pt, lo, hi = boot(arms[half][1], arms[rows][1], metric)
        seeds = sorted(set(arms[rows]) & set(arms[half]))
        deltas = [mean(arms[rows][s], metric) - mean(arms[half][s], metric) for s in seeds]
        spread = [x for x in (sd.get(rows), sd.get(half)) if x is not None]
        line = f"  full={rows:<5} half={half:<5} seed 1 {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        entry = {"full": rows, "half": half, "delta_seed1": round(pt, 4), "ci_seed1": [round(lo, 4), round(hi, 4)]}
        if len(seeds) > 1:
            d_mean, d_sd = statistics.fmean(deltas), statistics.stdev(deltas)
            line += f"  | {len(seeds)} seeds: {d_mean:+.4f} (SD {d_sd:.4f})"
            entry.update(seeds=seeds, deltas=[round(d, 4) for d in deltas], delta_mean=round(d_mean, 4), delta_sd=round(d_sd, 4))
            if spread:
                ratio = d_mean / statistics.fmean(spread)
                line += f", {ratio:.1f}x the seed SD of the arms"
                entry["x_seed_sd"] = round(ratio, 1)
        print(line)
        out.append(entry)
    if "--json" in sys.argv:
        means = {rows: {"seeds": len(runs),
                        **{m: round(statistics.fmean(mean(d, m) for d in runs.values()), 4) for m in SHOWN[task]},
                        f"{metric}_sd": None if sd[rows] is None else round(sd[rows], 4)}
                 for rows, runs in sorted(arms.items())}
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(
            json.dumps({"task": task, "curve": folder.name, "size": size, "metric": metric,
                        "arms": means, "denominators": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
