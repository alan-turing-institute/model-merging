# Data-parallel merging: CLINC150 intent classification (labels prompt), Gemma 3 4B

Generated 2026-11-05 by `evaluate/make_data_parallel_tables.py` from the `.eval` logs in `evaluate/logs/`.

The training data is split into N equal parts; an expert is trained on each part (LoRA, from google/gemma-3-4b-it) and the N experts are merged. Every arm is repeated with seeds 1-5 (training data order and initialisation, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once. Test set: the 4,500 test queries (30 per intent), with the prompt that lists all 150 labels; accuracy, and the share of answers that are one of the labels.

Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.

## N = 2

| | Accuracy | Valid label | seeds |
|---|---|---|---|
| Base model (untrained) | 0.704 | 0.984 | – |
| **Full data** (one model) | 0.976 ± 0.002 | 1.000 ± 0.000 | 5 |
| Experts (each on 1/2 of the data) | 0.971 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: linear | 0.974 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: ties | 0.975 ± 0.002 | 1.000 ± 0.000 | 5 |
| merge: dare_ties | 0.975 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: task_arithmetic | 0.974 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: model_stock | 0.949 ± 0.002 | 1.000 ± 0.000 | 5 |
| merge: slerp | 0.974 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: arcee_fusion | 0.974 ± 0.001 | 1.000 ± 0.000 | 5 |

## N = 4

| | Accuracy | Valid label | seeds |
|---|---|---|---|
| Base model (untrained) | 0.704 | 0.984 | – |
| **Full data** (one model) | 0.976 ± 0.002 | 1.000 ± 0.000 | 5 |
| Experts (each on 1/4 of the data) | 0.965 ± 0.001 | 0.999 ± 0.000 | 5 |
| merge: linear | 0.972 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: ties | 0.974 ± 0.002 | 1.000 ± 0.000 | 5 |
| merge: dare_ties | 0.972 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: task_arithmetic | 0.971 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: model_stock | 0.958 ± 0.002 | 1.000 ± 0.000 | 5 |
| merge: slerp | 0.971 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: arcee_fusion | 0.972 ± 0.001 | 1.000 ± 0.000 | 5 |

## N = 8

| | Accuracy | Valid label | seeds |
|---|---|---|---|
| Base model (untrained) | 0.704 | 0.984 | – |
| **Full data** (one model) | 0.976 ± 0.002 | 1.000 ± 0.000 | 5 |
| Experts (each on 1/8 of the data) | 0.953 ± 0.001 | 0.999 ± 0.000 | 5 |
| merge: linear | 0.964 ± 0.001 | 0.999 ± 0.000 | 5 |
| merge: ties | 0.966 ± 0.001 | 0.999 ± 0.001 | 5 |
| merge: dare_ties | 0.965 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: task_arithmetic | 0.964 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: model_stock | 0.957 ± 0.001 | 1.000 ± 0.000 | 5 |
| merge: slerp | 0.964 ± 0.001 | 0.999 ± 0.000 | 5 |
| merge: arcee_fusion | 0.963 ± 0.001 | 1.000 ± 0.000 | 5 |
