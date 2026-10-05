# Pre-registration: does a two-way merge beat the better of its two shards?

Written and committed **before** the experts for seeds 9–16 were trained, and
before any of their results existed. The point of the document is the timestamp.

## Why this run exists

An earlier arm answered the same question at eight seeds and found
`merge − best half = +0.0057`, 95% CI `[+0.0012, +0.0101]`, positive on seven of
eight. That interval clears zero, but the sample was not fixed in advance: five
seeds gave `[−0.0002, +0.0121]`, missing zero by two ten-thousandths, and three
further seeds were then trained. Extending a sample after seeing a near-miss
inflates the false-positive rate however the extension turns out, and a reviewer
is entitled to discount the result for that reason alone.

Nothing about the earlier estimate was changed by the extension — the mean moved
from +0.0059 to +0.0057 — but that is an argument, not a control. This run is
the control.

## Design, fixed here

- **Seeds 9 to 16**, eight of them, independent of the 1–8 used before. No seed
  from the earlier arm is reused.
- Each seed trains both crime half-experts (`crime_dataset1`, `crime_dataset2`,
  the stratified seed-42 split already in the repo), merges them linearly at
  weight 1/2, and scores all three on the same 1,077 held-out posts.
- **Primary estimand:** the mean of `merge − max(half1, half2)` paired by seed.
- **n = 8, fixed.** No seed will be added, whatever the outcome. If the interval
  straddles zero, that is the result.

## Decision rule, fixed here

The hypothesis is supported only if **both** hold:

1. the two-sided 95% confidence interval (t, 7 degrees of freedom) excludes zero; and
2. the mean exceeds 0.0039, the expert-draw spread measured on this task.

Anything else is reported as not supported. No subgroup, no alternative metric,
no re-analysis at a different n.

## Prediction

`+0.0057` (the earlier estimate). A materially smaller effect, or an interval
containing zero, means the earlier result owed something to the stopping rule.
