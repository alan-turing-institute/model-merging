# COMET-22 replication (Rei et al., 2022, WMT)

WMT21 News MQM. Segments / systems here: zh-en 9750 / 15, en-de 8959 / 17, en-ru 8432 / 16; paper: zh-en 9750 / 15, en-de 8959 / 17, en-ru 8432 / 16.

| metric | zh-en rho / tau | en-de rho / tau | en-ru rho / tau | avg rho / tau | zh-en sys acc | en-de sys acc | en-ru sys acc | avg acc |
|---|---|---|---|---|---|---|---|---|
| DA Estimator (= wmt22-comet-da) (paper) | 0.495 / 0.362 | 0.289 / 0.221 | 0.369 / 0.285 | 0.384 / 0.289 | 50.47 | 71.32 | 66.66 | 62.82 |
| COMET-22 (ensemble, not released) (paper) | 0.537 / 0.395 | 0.366 / 0.281 | 0.407 / 0.315 | 0.437 / 0.330 | 64.76 | 86.76 | 70.83 | 74.12 |
| BLEU (paper) | 0.215 / 0.153 | 0.086 / 0.065 | 0.123 / 0.094 | 0.141 / 0.104 | 45.71 | 66.91 | 46.66 | 53.09 |
| chrF (paper) | 0.116 / 0.088 | 0.116 / 0.088 | 0.213 / 0.165 | 0.148 / 0.114 | 43.81 | 65.44 | 55.00 | 54.75 |
| wmt22-comet-da (ours) | 0.493 / 0.360 | 0.296 / 0.227 | 0.368 / 0.285 | 0.386 / 0.290 | 50.48 | 77.94 | 71.67 | 66.69 |
| BLEU (ours) | 0.215 / 0.153 | 0.086 / 0.065 | 0.123 / 0.094 | 0.141 / 0.104 | 47.62 | 65.44 | 54.17 | 55.74 |
| chrF (ours) | 0.246 / 0.176 | 0.116 / 0.088 | 0.213 / 0.165 | 0.192 / 0.143 | 46.67 | 63.97 | 57.50 | 56.05 |

## Reading

- **Segment level: replicated.** wmt22-comet-da is within 0.007 of the paper's DA
  Estimator row in every language pair (average rho / tau 0.386 / 0.290 against
  0.384 / 0.289), so the model, its inputs (src, mt, ref) and our scaling are
  set up as the authors intended.
- **The control matches exactly.** Our BLEU equals the paper's in every cell, and
  chrF in en-de and en-ru, so the data (segment and system counts are the
  paper's) and the correlation code are the paper's too.
- **The paper's chrF zh-en cell looks like a copy of its en-de cell.** Its own
  average row (0.192 / 0.143) is the mean of our zh-en value (0.246 / 0.176)
  with its en-de and en-ru, not of the printed cell; the "chrF (paper)" average
  above is computed from the printed cells.
- **System-level pairwise accuracy is not reproduced exactly, for any metric.**
  zh-en COMET matches (50.48 vs 50.47), en-de and en-ru come out 5-7 points
  higher (77.94 / 71.67 vs 71.32 / 66.66), and BLEU differs too (47.6 / 65.4 /
  54.2 vs 45.7 / 66.9 / 46.7). The difference is in the system-level protocol -
  how system scores are aggregated, which systems (human translations) are in,
  ties - not in COMET; with 105-136 system pairs, one or two flipped pairs move
  accuracy by about 1-2 points.
