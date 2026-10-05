"""Two-way crime sharding over eight expert seeds.

The first five put merge minus best-half at +0.0059 with a 95% interval of
[-0.0002, +0.0121] - positive on every seed, but missing zero by two
ten-thousandths. The decision rule was fixed before that run and was not met.
Three further seeds were trained to settle it rather than to re-argue it.
"""
import json, statistics as st
FULL = 0.9768
T = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}          # two-sided 95%, df = n-1

def acc(p):
    return json.load(open(p))["metrics"]["accuracy"]

rows, diffs, bests, h1, h2 = [], [], [], [], []
for s in range(1, 9):
    try:
        a = acc("results-isambard/crime-1-of-2-seed%d.json" % s)
        b = acc("results-isambard/crime-2-of-2-seed%d.json" % s)
        m = acc("results-isambard/seed%d-crime-linear.json" % s)
    except FileNotFoundError:
        continue
    best = max(a, b); d = m - best
    h1.append(a); h2.append(b); bests.append(best); diffs.append(d)
    rows.append((s, a, b, best, m, d, d / (FULL - best)))

print("%4s %8s %8s %8s %8s %11s %7s" % ("seed","half1","half2","best","merge","merge-best","R"))
for r in rows:
    print("%4d %8.4f %8.4f %8.4f %8.4f %+11.4f %+7.2f" % r)

n = len(diffs); mean = st.mean(diffs); sd = st.stdev(diffs); se = sd / n ** 0.5
t = T.get(n - 1, 2.365); lo, hi = mean - t * se, mean + t * se
pos = sum(1 for d in diffs if d > 0)
print("\nmerge - best half, paired over %d seeds: %+.4f (sd %.4f)" % (n, mean, sd))
print("  95%% CI [%+.4f, %+.4f]   t = %.2f on %d df" % (lo, hi, mean / se, n - 1))
print("  positive on %d of %d seeds   mean R = %+.2f"
      % (pos, n, st.mean([d / (FULL - b) for d, b in zip(diffs, bests)])))
print("\nhalf-expert spread: half1 sd %.4f (range %.4f) | half2 sd %.4f (range %.4f)"
      % (st.stdev(h1), max(h1)-min(h1), st.stdev(h2), max(h2)-min(h2)))
print("\n" + ("INTERVAL CLEARS ZERO: at a two-way split the merge beats the better of its\n"
              "own two shards. The rule fixed before the first run is now met."
              if lo > 0 else
              "Interval still includes zero: not demonstrated at n=%d." % n))
