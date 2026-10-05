"""Pre-registered replication, seeds 9-16 only. Rule fixed in
PREREGISTRATION-twoway.md before any of this data existed:
  (1) the two-sided 95% interval must exclude zero, and
  (2) the mean must exceed 0.0039, the expert-draw spread.
n = 8, no extension, whatever the outcome.
"""
import json, statistics as st
FULL = 0.9768
def acc(p): return json.load(open(p))["metrics"]["accuracy"]

rows, diffs = [], []
for s in range(9, 17):
    a = acc("results-isambard/crime-1-of-2-seed%d.json" % s)
    b = acc("results-isambard/crime-2-of-2-seed%d.json" % s)
    m = acc("results-isambard/seed%d-crime-linear.json" % s)
    best = max(a, b); d = m - best
    rows.append((s, a, b, best, m, d, d / (FULL - best))); diffs.append(d)

print("%4s %8s %8s %8s %8s %11s %7s" % ("seed","half1","half2","best","merge","merge-best","R"))
for r in rows: print("%4d %8.4f %8.4f %8.4f %8.4f %+11.4f %+7.2f" % r)

n=len(diffs); mean=st.mean(diffs); sd=st.stdev(diffs); se=sd/n**0.5; t=2.365  # 7 df
lo, hi = mean - t*se, mean + t*se
pos=sum(1 for d in diffs if d>0)
print("\nPRE-REGISTERED RESULT, seeds 9-16, n = %d" % n)
print("  merge - best half: %+.4f (sd %.4f)   95%% CI [%+.4f, %+.4f]   t = %.2f" % (mean, sd, lo, hi, mean/se))
print("  positive on %d of %d   mean R = %+.2f" % (pos, n, st.mean([r[6] for r in rows])))
c1 = lo > 0; c2 = mean > 0.0039
print("\n  condition 1, interval excludes zero : %s" % ("MET" if c1 else "NOT MET"))
print("  condition 2, mean exceeds 0.0039    : %s" % ("MET" if c2 else "NOT MET"))
print("\n  %s" % ("SUPPORTED: at a two-way split the merge beats the better of its own two shards."
                  if c1 and c2 else
                  "NOT SUPPORTED under the pre-registered rule."))
print("\n  earlier (seeds 1-8, extended sample): +0.0057 [+0.0012, +0.0101]")
print("  prediction recorded before this ran : +0.0057")
