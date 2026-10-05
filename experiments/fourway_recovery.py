"""Does a four-way merge beat the best single shard, reliably?

THE QUESTION. If data cannot be centralised, the alternative to merging is to
keep the better shard. So the comparison that decides whether merging is a
cheaper substitute for centralised training is merge minus BEST SHARD, paired by
expert seed - not merge against full-data training, which flatters merging by
comparing it to something the design has ruled out.

At two-way that difference was +0.0037 from a single pair, against an expert-draw
standard deviation of 0.0039: indistinguishable from zero. This runs it at
four-way, where the gap to recover is larger, over five paired seeds.

    uv run --project evaluate python experiments/fourway_recovery.py RESULTS_DIR

Reports the paired mean, its interval, and the recovery fraction. A positive
interval that clears zero is what "feasible" would require; anything else leaves
the two-way verdict standing.
"""
import json
import statistics as st
import sys
from pathlib import Path

FULL = 0.9768          # full-data crime model, same 1,077-post test set
SEED_SPREAD = 0.0039   # expert-draw s.d. measured on the five-seed half study


def load(path):
    m = json.load(open(path))["metrics"]
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    return m["accuracy"], tp / (tp + fn), tn / (tn + fp)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results-isambard")
    rows, diffs = [], []
    print(f"{'seed':>4}  {'q1':>7} {'q2':>7} {'q3':>7} {'q4':>7}  "
          f"{'best':>7} {'merge':>7}  {'merge-best':>11}  {'R':>6}")
    for seed in range(1, 6):
        qs = []
        for q in range(1, 5):
            p = root / f"crime-q{q}-seed{seed}.json"
            if not p.exists():
                print(f"{seed:>4}  missing {p.name}"); break
            qs.append(load(p))
        else:
            mp = root / f"crime-4way-merge-seed{seed}.json"
            if not mp.exists():
                print(f"{seed:>4}  missing {mp.name}"); continue
            merge = load(mp)
            best = max(a for a, _, _ in qs)
            d = merge[0] - best
            R = d / (FULL - best) if FULL > best else float("nan")
            diffs.append(d)
            rows.append((seed, qs, best, merge, d, R))
            print(f"{seed:>4}  " + " ".join(f"{a:7.4f}" for a, _, _ in qs) +
                  f"  {best:7.4f} {merge[0]:7.4f}  {d:+11.4f}  {R:+6.2f}")

    if len(diffs) < 2:
        raise SystemExit("\nnot enough seeds yet for a paired interval")

    n = len(diffs)
    mean, sd = st.fmean(diffs), st.stdev(diffs)
    se = sd / n ** 0.5
    t_crit = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776}.get(n - 1, 2.776)
    lo, hi = mean - t_crit * se, mean + t_crit * se
    print(f"\nmerge - best shard, paired over {n} seeds: {mean:+.4f} "
          f"(sd {sd:.4f})  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"mean recovery fraction R = {st.fmean(r for *_, r in rows):+.2f}")

    if lo > 0 and mean > SEED_SPREAD:
        print("\nThe interval clears zero and the effect exceeds the expert-draw spread:\n"
              "merging beats keeping the best shard at four-way on this task.")
    elif lo > 0:
        print(f"\nThe interval clears zero but the effect ({mean:+.4f}) is within the\n"
              f"expert-draw spread ({SEED_SPREAD}); report it as suggestive, not settled.")
    else:
        print("\nThe interval includes zero: merging is not shown to beat keeping the\n"
              "best single shard, and the two-way verdict stands at four-way too.")
    # Sensitivity and specificity, because every degradation measured on this
    # task has been one-directional - arms lose accuracy by missing crime posts.
    print(f"\n{'seed':>4}  {'best-shard sens/spec':>22}  {'merge sens/spec':>22}")
    for seed, qs, best, merge, _, _ in rows:
        bq = max(qs, key=lambda x: x[0])
        print(f"{seed:>4}  {bq[1]:10.4f}/{bq[2]:<11.4f}  {merge[1]:10.4f}/{merge[2]:<11.4f}")


if __name__ == "__main__":
    main()
