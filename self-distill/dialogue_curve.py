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
    arms = {}
    for p in folder.glob(f"*{size}.jsonl"):
        arm = p.stem[: -len(size)]
        rows = 0 if arm == "base" else int(re.fullmatch(r"n(\d+)", arm).group(1))
        arms[rows] = (arm, load(p))
    print(f"{folder.name}{size}: {len(next(iter(arms.values()))[1])} test dialogues")
    print(f"{'rows':>6}  " + "".join(f"{m:>12}" for m in SHOWN[task]))
    for rows in sorted(arms):
        data = arms[rows][1]
        print(f"{rows:>6}  " + "".join(f"{statistics.fmean(v[m] for v in data.values()):>12.3f}" for m in SHOWN[task]))
    print(f"denominator for a two-way split ({metric}, full minus half):")
    out = []
    for rows in sorted(arms):
        if rows and rows // 2 in arms and rows // 2:
            pt, lo, hi = boot(arms[rows // 2][1], arms[rows][1], metric)
            out.append({"full": rows, "half": rows // 2, "delta": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)]})
            print(f"  full={rows:<5} half={rows // 2:<5} {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    if "--json" in sys.argv:
        means = {rows: {m: round(statistics.fmean(v[m] for v in d.values()), 4) for m in SHOWN[task]}
                 for rows, (_, d) in sorted(arms.items())}
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(
            json.dumps({"task": task, "curve": folder.name, "size": size, "metric": metric, "arms": means, "denominators": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
