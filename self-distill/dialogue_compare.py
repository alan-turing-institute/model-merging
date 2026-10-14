"""Paired comparison of two dialogue-task runs on the same test dialogues.

    python dialogue_compare.py ../results/dialogue_curves/reorder kd27b_ce50_1b sftloop_1b [--metric tau]

Prints each run's mean and the paired bootstrap 95% interval of A minus B.
"""
import json
import random
import statistics
import sys
from pathlib import Path

folder, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
metric = sys.argv[sys.argv.index("--metric") + 1] if "--metric" in sys.argv else "tau"
load = lambda n: {r["id"]: float(r["check"][metric]) for r in map(json.loads, (folder / f"{n}.jsonl").read_text().splitlines())}
x, y = load(a), load(b)
ids = sorted(set(x) & set(y)); d = [x[i] - y[i] for i in ids]
rng = random.Random(0)
boots = sorted(statistics.fmean(d[rng.randrange(len(d))] for _ in d) for _ in range(2000))
print(f"{a} {statistics.fmean(x[i] for i in ids):.4f} vs {b} {statistics.fmean(y[i] for i in ids):.4f}: "
      f"{statistics.fmean(d):+.4f} [{boots[50]:+.4f}, {boots[1949]:+.4f}] n={len(ids)} ({metric})")
