"""Recovery fraction R for the data-parallel reorder merges (Gemma 3 1B).

    python dialogue_merge_report.py ../results/dialogue_curves/reorder _1b [--json out.json]

Per seed: full = n4000, expert A = n2000 (the first half), expert B = half_b
(the second half), merges = merged-<method>. R = (merged - best expert) /
(full - best expert) on Kendall tau over the 518 test dialogues, with a
paired bootstrap 95% interval over dialogues.
"""
import json
import random
import statistics
import sys
from pathlib import Path

METHODS = ["linear", "ties", "dare_ties", "task_arithmetic"]


def load(path):
    return {r["id"]: float(r["check"]["tau"]) for r in map(json.loads, path.read_text().splitlines())}


def main():
    folder, size = Path(sys.argv[1]), sys.argv[2]
    get = lambda stem: load(folder / f"{stem}{size}.jsonl") if (folder / f"{stem}{size}.jsonl").exists() else None
    mean = lambda d: statistics.fmean(d.values())
    rng, out = random.Random(0), []
    print(f"{'method':16} {'seed':4} {'merged':>7} {'exp A':>7} {'exp B':>7} {'full':>7} {'R':>7}  95% CI")
    for method in METHODS:
        for s in ("", "_s2", "_s3"):
            m, a, b, f = get(f"merged-{method}{s}"), get(f"n2000{s}"), get(f"half_b{s}"), get(f"n4000{s}")
            if not (m and a and b and f):
                continue
            best = a if mean(a) >= mean(b) else b
            ids = sorted(set(m) & set(best) & set(f))
            r = lambda idx: (sum(m[ids[i]] - best[ids[i]] for i in idx)) / (sum(f[ids[i]] - best[ids[i]] for i in idx))
            point = r(range(len(ids)))
            boots = sorted(r([rng.randrange(len(ids)) for _ in ids]) for _ in range(2000))
            seed = s.strip("_s") or "1"
            print(f"{method:16} {seed:4} {mean(m):7.3f} {mean(a):7.3f} {mean(b):7.3f} {mean(f):7.3f} {point:+7.3f}  [{boots[50]:+.3f}, {boots[1949]:+.3f}]")
            out.append({"method": method, "seed": seed, "merged": mean(m), "expert_a": mean(a), "expert_b": mean(b),
                        "full": mean(f), "R": point, "ci": [boots[50], boots[1949]]})
    for method in METHODS:
        rs = [x["R"] for x in out if x["method"] == method]
        if len(rs) > 1:
            print(f"{method:16} mean R over {len(rs)} seeds {statistics.fmean(rs):+.3f}, SD {statistics.stdev(rs):.3f}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
