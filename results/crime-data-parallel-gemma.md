# Data-parallel merging: Crime classification (Reddit posts), Gemma 3 4B

Generated 2026-11-03 by `evaluate/make_data_parallel_tables.py` from the `.eval` logs in `evaluate/logs/`.

The training data is split into N equal parts; an expert is trained on each part (LoRA for crime, self-distillation for XSum) and the N experts are merged. Every arm is repeated with seeds 1-5 (training data order and initialisation, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once. Test set: the 1,077 test posts; accuracy and F1 for the crime class.

Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.

## N = 2

| | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.755 | 0.624 | – |
| **Full data** (one model) | 0.981 ± 0.002 | 0.978 ± 0.002 | 5 |
| Experts (each on 1/2 of the data) | 0.960 ± 0.003 | 0.954 ± 0.004 | 5 |
| merge: linear | 0.965 ± 0.005 | 0.960 ± 0.006 | 5 |
| merge: ties | 0.968 ± 0.007 | 0.964 ± 0.008 | 5 |
| merge: dare_ties | 0.965 ± 0.005 | 0.961 ± 0.006 | 5 |
| merge: task_arithmetic | 0.966 ± 0.004 | 0.961 ± 0.005 | 5 |
| merge: model_stock | 0.910 ± 0.010 | 0.889 ± 0.014 | 5 |
| merge: slerp | 0.966 ± 0.004 | 0.961 ± 0.005 | 5 |
| merge: arcee_fusion | 0.961 ± 0.006 | 0.955 ± 0.008 | 5 |

## N = 4

| | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.755 | 0.624 | – |
| **Full data** (one model) | 0.981 ± 0.002 | 0.978 ± 0.002 | 5 |
| Experts (each on 1/4 of the data) | 0.939 ± 0.002 | 0.930 ± 0.002 | 5 |
| merge: linear | 0.945 ± 0.007 | 0.936 ± 0.008 | 5 |
| merge: ties | 0.957 ± 0.003 | 0.951 ± 0.004 | 5 |
| merge: dare_ties | 0.945 ± 0.005 | 0.937 ± 0.007 | 5 |
| merge: task_arithmetic | 0.945 ± 0.006 | 0.936 ± 0.008 | 5 |
| merge: model_stock | 0.918 ± 0.009 | 0.901 ± 0.013 | 5 |
| merge: slerp | 0.945 ± 0.006 | 0.937 ± 0.008 | 5 |
| merge: arcee_fusion | 0.943 ± 0.005 | 0.934 ± 0.007 | 5 |

## N = 8

| | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.755 | 0.624 | – |
| **Full data** (one model) | 0.981 ± 0.002 | 0.978 ± 0.002 | 5 |
| Experts (each on 1/8 of the data) | 0.917 ± 0.002 | 0.905 ± 0.002 | 5 |
| merge: linear | 0.917 ± 0.003 | 0.903 ± 0.004 | 5 |
| merge: ties | 0.935 ± 0.006 | 0.927 ± 0.007 | 5 |
| merge: dare_ties | 0.919 ± 0.002 | 0.905 ± 0.003 | 5 |
| merge: task_arithmetic | 0.918 ± 0.003 | 0.903 ± 0.004 | 5 |
| merge: model_stock | 0.905 ± 0.004 | 0.886 ± 0.005 | 5 |
| merge: slerp | 0.918 ± 0.002 | 0.904 ± 0.003 | 5 |
| merge: arcee_fusion | 0.918 ± 0.006 | 0.903 ± 0.008 | 5 |
