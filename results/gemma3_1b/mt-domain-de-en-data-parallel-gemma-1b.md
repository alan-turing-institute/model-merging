# Data-parallel merging: German-English translation (medical and legal), Gemma 3 1B

Generated 2026-11-05 by `evaluate/make_data_parallel_tables.py` from the `.eval` logs in `evaluate/logs-1b/`.

The training data is split into N equal parts; an expert is trained on each part (LoRA, from google/gemma-3-1b-it) and the N experts are merged. Every arm is repeated with seeds 1-5 (training data order and initialisation, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once. Test set: the EMEA and JRC-Acquis test pairs; mean sentence chrF++, overall and per domain, and output length over reference length in words.

Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.

## N = 2

| | chrF++ | Medical | Legal | Length ratio | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 32.41 | 33.26 | 31.57 | 6.76 | – |
| **Full data** (one model) | 57.83 ± 0.17 | 59.31 ± 0.26 | 56.35 ± 0.09 | 1.13 ± 0.08 | 5 |
| Experts (each on 1/2 of the data) | 55.89 ± 0.08 | 57.61 ± 0.15 | 54.16 ± 0.20 | 1.09 ± 0.04 | 5 |
| merge: linear | 56.43 ± 0.16 | 58.14 ± 0.15 | 54.72 ± 0.21 | 1.06 ± 0.03 | 5 |
| merge: ties | 55.74 ± 0.32 | 57.39 ± 0.36 | 54.09 ± 0.38 | 1.09 ± 0.03 | 5 |
| merge: dare_ties | 56.28 ± 0.23 | 58.03 ± 0.22 | 54.53 ± 0.38 | 1.06 ± 0.02 | 5 |
| merge: task_arithmetic | 56.45 ± 0.19 | 58.20 ± 0.22 | 54.70 ± 0.19 | 1.06 ± 0.03 | 5 |
| merge: model_stock | 55.37 ± 0.15 | 57.06 ± 0.13 | 53.69 ± 0.22 | 1.03 ± 0.01 | 5 |
| merge: slerp | 56.43 ± 0.15 | 58.13 ± 0.22 | 54.74 ± 0.18 | 1.05 ± 0.02 | 5 |
| merge: arcee_fusion | 56.21 ± 0.22 | 57.98 ± 0.29 | 54.43 ± 0.24 | 1.07 ± 0.02 | 5 |

## N = 4

| | chrF++ | Medical | Legal | Length ratio | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 32.41 | 33.26 | 31.57 | 6.76 | – |
| **Full data** (one model) | 57.83 ± 0.17 | 59.31 ± 0.26 | 56.35 ± 0.09 | 1.13 ± 0.08 | 5 |
| Experts (each on 1/4 of the data) | 54.06 ± 0.06 | 55.73 ± 0.11 | 52.39 ± 0.12 | 1.16 ± 0.05 | 5 |
| merge: linear | 54.65 ± 0.11 | 56.31 ± 0.07 | 52.99 ± 0.28 | 1.12 ± 0.08 | 5 |
| merge: ties | 53.94 ± 0.25 | 55.83 ± 0.41 | 52.05 ± 0.18 | 1.15 ± 0.08 | 5 |
| merge: dare_ties | 54.56 ± 0.17 | 56.12 ± 0.16 | 53.00 ± 0.28 | 1.12 ± 0.09 | 5 |
| merge: task_arithmetic | 54.60 ± 0.15 | 56.28 ± 0.09 | 52.91 ± 0.24 | 1.12 ± 0.08 | 5 |
| merge: model_stock | 54.55 ± 0.17 | 56.15 ± 0.24 | 52.94 ± 0.26 | 1.03 ± 0.01 | 5 |
| merge: slerp | 54.66 ± 0.20 | 56.34 ± 0.17 | 52.98 ± 0.28 | 1.13 ± 0.10 | 5 |
| merge: arcee_fusion | 54.59 ± 0.12 | 56.32 ± 0.25 | 52.85 ± 0.12 | 1.17 ± 0.08 | 5 |

## N = 8

| | chrF++ | Medical | Legal | Length ratio | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 32.41 | 33.26 | 31.57 | 6.76 | – |
| **Full data** (one model) | 57.83 ± 0.17 | 59.31 ± 0.26 | 56.35 ± 0.09 | 1.13 ± 0.08 | 5 |
| Experts (each on 1/8 of the data) | 52.35 ± 0.10 | 53.91 ± 0.09 | 50.80 ± 0.12 | 1.21 ± 0.02 | 5 |
| merge: linear | 52.79 ± 0.11 | 54.46 ± 0.15 | 51.13 ± 0.12 | 1.18 ± 0.07 | 5 |
| merge: ties | 51.80 ± 0.19 | 53.63 ± 0.24 | 49.97 ± 0.24 | 1.42 ± 0.19 | 5 |
| merge: dare_ties | 52.68 ± 0.10 | 54.32 ± 0.09 | 51.05 ± 0.15 | 1.15 ± 0.07 | 5 |
| merge: task_arithmetic | 52.79 ± 0.13 | 54.48 ± 0.09 | 51.10 ± 0.22 | 1.15 ± 0.07 | 5 |
| merge: model_stock | 53.03 ± 0.05 | 54.60 ± 0.14 | 51.46 ± 0.07 | 1.07 ± 0.03 | 5 |
| merge: slerp | 52.79 ± 0.10 | 54.50 ± 0.15 | 51.07 ± 0.08 | 1.16 ± 0.08 | 5 |
| merge: arcee_fusion | 52.56 ± 0.26 | 54.17 ± 0.24 | 50.95 ± 0.31 | 1.14 ± 0.05 | 5 |
