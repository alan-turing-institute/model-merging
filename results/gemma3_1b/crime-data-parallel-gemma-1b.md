# Data-parallel merging: Crime classification (Reddit posts), Gemma 3 1B

Generated 2026-11-05 by `evaluate/make_data_parallel_tables.py` from the `.eval` logs in `evaluate/logs-1b/`.

The training data is split into N equal parts; an expert is trained on each part (LoRA, from google/gemma-3-1b-it) and the N experts are merged. Every arm is repeated with seeds 1-5 (training data order and initialisation, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once. Test set: the 1,077 test posts; accuracy and F1 for the crime class.

Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.

## N = 2

| | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.664 | 0.395 | – |
| **Full data** (one model) | 0.971 ± 0.003 | 0.968 ± 0.004 | 5 |
| Experts (each on 1/2 of the data) | 0.951 ± 0.003 | 0.944 ± 0.004 | 5 |
| merge: linear | 0.950 ± 0.002 | 0.943 ± 0.003 | 5 |
| merge: ties | 0.955 ± 0.006 | 0.949 ± 0.007 | 5 |
| merge: dare_ties | 0.951 ± 0.003 | 0.945 ± 0.003 | 5 |
| merge: task_arithmetic | 0.951 ± 0.003 | 0.944 ± 0.003 | 5 |
| merge: model_stock | 0.930 ± 0.004 | 0.918 ± 0.006 | 5 |
| merge: slerp | 0.951 ± 0.003 | 0.944 ± 0.004 | 5 |
| merge: arcee_fusion | 0.952 ± 0.002 | 0.945 ± 0.003 | 5 |

## N = 4

| | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.664 | 0.395 | – |
| **Full data** (one model) | 0.971 ± 0.003 | 0.968 ± 0.004 | 5 |
| Experts (each on 1/4 of the data) | 0.926 ± 0.002 | 0.916 ± 0.002 | 5 |
| merge: linear | 0.933 ± 0.003 | 0.925 ± 0.003 | 5 |
| merge: ties | 0.933 ± 0.008 | 0.925 ± 0.008 | 5 |
| merge: dare_ties | 0.933 ± 0.003 | 0.925 ± 0.004 | 5 |
| merge: task_arithmetic | 0.933 ± 0.004 | 0.924 ± 0.004 | 5 |
| merge: model_stock | 0.926 ± 0.003 | 0.915 ± 0.003 | 5 |
| merge: slerp | 0.933 ± 0.005 | 0.924 ± 0.005 | 5 |
| merge: arcee_fusion | 0.930 ± 0.004 | 0.921 ± 0.003 | 5 |

## N = 8

| | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.664 | 0.395 | – |
| **Full data** (one model) | 0.971 ± 0.003 | 0.968 ± 0.004 | 5 |
| Experts (each on 1/8 of the data) | 0.899 ± 0.003 | 0.885 ± 0.003 | 5 |
| merge: linear | 0.907 ± 0.001 | 0.895 ± 0.002 | 5 |
| merge: ties | 0.914 ± 0.004 | 0.906 ± 0.005 | 5 |
| merge: dare_ties | 0.907 ± 0.001 | 0.895 ± 0.002 | 5 |
| merge: task_arithmetic | 0.906 ± 0.002 | 0.894 ± 0.003 | 5 |
| merge: model_stock | 0.903 ± 0.002 | 0.890 ± 0.003 | 5 |
| merge: slerp | 0.906 ± 0.001 | 0.895 ± 0.002 | 5 |
| merge: arcee_fusion | 0.907 ± 0.004 | 0.894 ± 0.006 | 5 |
