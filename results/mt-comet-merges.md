# COMET-22 on the EN->DE merge experiments

`Unbabel/wmt22-comet-da`, mean segment score x 100; chrF++ from the same logs.

## Data-parallel, 4B (domain experts: medical vs law)

| model | COMET | COMET medical | COMET law | chrF++ | runs |
|---|---|---|---|---|---|
| untrained | 45.56 | 45.67 | 45.46 | 18.86 | 1 |
| full data | 83.59 | 82.45 | 84.73 | 59.54 | 1 |
| IID half | 83.59 | 82.53 | 84.65 | 58.30 | 1 |
| expert: medical | 80.75 ±0.10 | 82.44 | 79.07 | 54.42 | 3 |
| expert: law | 83.41 ±0.03 | 81.91 | 84.90 | 57.08 | 3 |
| better expert | 83.41 ±0.03 | 81.91 | 84.90 | 57.08 | 3 |
| merged: linear | 84.02 ±0.06 | 83.03 | 85.00 | 58.59 | 3 |
| merged: ties | 81.81 ±0.30 | 82.33 | 81.28 | 56.22 | 3 |
| merged: dare_ties | 83.93 ±0.07 | 82.98 | 84.89 | 58.46 | 3 |
| merged: task_arithmetic | 84.00 ±0.10 | 83.02 | 84.98 | 58.60 | 3 |

| method | seed | merged | better expert | full | R | 95% CI |
|---|---|---|---|---|---|---|
| linear | 1 | 84.05 | 83.43 | 83.59 | +3.764 | [-34.802, +27.075] |
| linear | 2 | 84.05 | 83.42 | 83.59 | +3.700 | [-30.991, +38.791] |
| linear | 3 | 83.95 | 83.37 | 83.59 | +2.645 | [-16.543, +20.551] |
| ties | 1 | 81.58 | 83.43 | 83.59 | -11.197 | [-119.579, +82.313] |
| ties | 2 | 82.14 | 83.42 | 83.59 | -7.553 | [-82.208, +68.305] |
| ties | 3 | 81.70 | 83.37 | 83.59 | -7.629 | [-73.780, +51.544] |
| dare_ties | 1 | 83.99 | 83.43 | 83.59 | +3.397 | [-27.600, +31.442] |
| dare_ties | 2 | 83.95 | 83.42 | 83.59 | +3.132 | [-26.135, +30.355] |
| dare_ties | 3 | 83.86 | 83.37 | 83.59 | +2.211 | [-9.901, +14.955] |
| task_arithmetic | 1 | 84.02 | 83.43 | 83.59 | +3.598 | [-27.697, +34.327] |
| task_arithmetic | 2 | 84.08 | 83.42 | 83.59 | +3.908 | [-32.644, +38.815] |
| task_arithmetic | 3 | 83.89 | 83.37 | 83.59 | +2.355 | [-17.471, +20.251] |
| linear | mean | | | | +3.370 (SD 0.629) | |
| ties | mean | | | | -8.793 (SD 2.083) | |
| dare_ties | mean | | | | +2.913 (SD 0.623) | |
| task_arithmetic | mean | | | | +3.287 (SD 0.822) | |

## Data-parallel, 1B (no 1B merges yet)

| model | COMET | COMET medical | COMET law | chrF++ | runs |
|---|---|---|---|---|---|
| untrained | 44.57 | 45.04 | 44.11 | 19.18 | 1 |
| full data | 80.26 | 80.23 | 80.30 | 53.70 | 1 |
| expert: medical | 76.58 | 79.53 | 73.63 | 47.95 | 1 |
| expert: law | 79.12 | 77.91 | 80.33 | 50.27 | 1 |

## Task-parallel merges, scored on EN->DE

Retention = (score - untrained) / (EN->DE specialist - untrained); the interval is merged minus specialist.

### clinc150-mt_domain, 4b

| model | COMET | retention | chrF++ | merged minus specialist [95% CI] |
|---|---|---|---|---|
| untrained | 45.56 | 0.000 | 18.86 |  |
| EN->DE specialist | 83.59 | 1.000 | 59.54 |  |
| multi-task ceiling | 83.62 | 1.001 | 59.59 |  |
| merged: linear | 83.89 | 1.008 | 57.92 | +0.30 [-0.01, +0.60] |
| merged: ties | 83.13 | 0.988 | 57.25 | -0.46 [-0.80, -0.14] |
| merged: dare_ties | 83.89 | 1.008 | 57.64 | +0.30 [-0.00, +0.62] |
| merged: task_arithmetic | 83.92 | 1.009 | 57.96 | +0.33 [+0.03, +0.66] |

### mt_domain-xsumc, 4b

| model | COMET | retention | chrF++ | merged minus specialist [95% CI] |
|---|---|---|---|---|
| untrained | 45.56 | 0.000 | 18.86 |  |
| EN->DE specialist | 83.59 | 1.000 | 59.54 |  |
| multi-task ceiling | 83.80 | 1.006 | 59.83 |  |
| merged: linear | 84.25 | 1.017 | 58.82 | +0.66 [+0.38, +0.95] |
| merged: ties | 83.99 | 1.010 | 59.56 | +0.40 [+0.18, +0.63] |
| merged: dare_ties | 84.22 | 1.016 | 58.78 | +0.62 [+0.34, +0.91] |
| merged: task_arithmetic | 84.27 | 1.018 | 58.83 | +0.68 [+0.40, +0.95] |

### clinc150-mt_domain-xsumc, 4b

| model | COMET | retention | chrF++ | merged minus specialist [95% CI] |
|---|---|---|---|---|
| untrained | 45.56 | 0.000 | 18.86 |  |
| EN->DE specialist | 83.59 | 1.000 | 59.54 |  |
| multi-task ceiling | 83.81 | 1.006 | 59.60 |  |
| merged: linear | 83.75 | 1.004 | 56.60 | +0.16 [-0.17, +0.49] |
| merged: ties | 83.34 | 0.993 | 57.26 | -0.25 [-0.57, +0.07] |
| merged: dare_ties | 83.73 | 1.004 | 56.71 | +0.14 [-0.18, +0.47] |
| merged: task_arithmetic | 83.73 | 1.004 | 56.51 | +0.14 [-0.20, +0.49] |

### clinc150-mt_domain, 1b

| model | COMET | retention | chrF++ | merged minus specialist [95% CI] |
|---|---|---|---|---|
| untrained | 44.57 | 0.000 | 19.18 |  |
| EN->DE specialist | 80.26 | 1.000 | 53.70 |  |
| multi-task ceiling | 80.27 | 1.000 | 54.13 |  |
| merged: linear | 80.01 | 0.993 | 50.61 | -0.25 [-0.63, +0.13] |
| merged: ties | 78.92 | 0.962 | 50.42 | -1.35 [-1.74, -0.98] |
| merged: dare_ties | 79.64 | 0.983 | 50.01 | -0.62 [-0.99, -0.24] |
| merged: task_arithmetic | 80.05 | 0.994 | 50.54 | -0.21 [-0.58, +0.15] |

### mt_domain-xsumc, 1b

| model | COMET | retention | chrF++ | merged minus specialist [95% CI] |
|---|---|---|---|---|
| untrained | 44.57 | 0.000 | 19.18 |  |
| EN->DE specialist | 80.26 | 1.000 | 53.70 |  |
| multi-task ceiling | 80.09 | 0.995 | 53.28 |  |
| merged: linear | 80.09 | 0.995 | 50.78 | -0.17 [-0.54, +0.20] |
| merged: ties | 80.21 | 0.999 | 51.86 | -0.05 [-0.39, +0.29] |
| merged: dare_ties | 80.15 | 0.997 | 50.80 | -0.12 [-0.48, +0.25] |
| merged: task_arithmetic | 80.25 | 1.000 | 50.77 | -0.01 [-0.36, +0.33] |

### clinc150-mt_domain-xsumc, 1b

| model | COMET | retention | chrF++ | merged minus specialist [95% CI] |
|---|---|---|---|---|
| untrained | 44.57 | 0.000 | 19.18 |  |
| EN->DE specialist | 80.26 | 1.000 | 53.70 |  |
| multi-task ceiling | 79.90 | 0.990 | 53.56 |  |
| merged: linear | 79.33 | 0.974 | 48.59 | -0.93 [-1.34, -0.52] |
| merged: ties | 78.59 | 0.953 | 48.95 | -1.67 [-2.08, -1.30] |
| merged: dare_ties | 79.01 | 0.965 | 48.16 | -1.26 [-1.69, -0.82] |
| merged: task_arithmetic | 79.36 | 0.975 | 48.49 | -0.91 [-1.31, -0.51] |

## Gold-target distillation (CE weight), 4B

| model | COMET | COMET medical | COMET law | chrF++ | runs |
|---|---|---|---|---|---|
| gemma3-mt_domain-full | 83.59 | 82.45 | 84.73 | 59.54 | 1 |
| kdgold_ce100 | 83.88 | 82.87 | 84.88 | 59.89 | 1 |
| kdgold_ce90 | 84.10 | 82.87 | 85.32 | 60.12 | 1 |
| kdgold_ce75 | 84.17 | 83.11 | 85.22 | 60.05 | 1 |
| kdgold_ce50 | 83.99 | 82.96 | 85.02 | 58.44 | 1 |
