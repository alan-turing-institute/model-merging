# Data-parallel merging: Wine review points (80-100), Gemma 3 1B

Generated 2026-11-05 by `evaluate/make_data_parallel_tables.py` from the `.eval` logs in `evaluate/logs-1b/`.

The training data is split into N equal parts; an expert is trained on each part (LoRA, from google/gemma-3-1b-it) and the N experts are merged. Every arm is repeated with seeds 1-5 (training data order and initialisation, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once. Test set: the 1,000 test reviews; mean absolute and root-mean-square error in points and Pearson correlation over the answered reviews, and the share with no number in the answer.

Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.

## N = 2

| | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 13.130 | 13.977 | 0.596 | 0.000 | – |
| **Full data** (one model) | 1.306 ± 0.011 | 1.723 ± 0.016 | 0.844 ± 0.001 | 0.000 ± 0.000 | 5 |
| Experts (each on 1/2 of the data) | 1.365 ± 0.022 | 1.815 ± 0.025 | 0.829 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: linear | 1.350 ± 0.023 | 1.789 ± 0.030 | 0.832 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: ties | 1.314 ± 0.020 | 1.737 ± 0.027 | 0.837 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: dare_ties | 1.349 ± 0.035 | 1.785 ± 0.043 | 0.834 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: task_arithmetic | 1.349 ± 0.019 | 1.788 ± 0.026 | 0.833 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: model_stock | 1.588 ± 0.024 | 2.073 ± 0.028 | 0.815 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: slerp | 1.351 ± 0.019 | 1.790 ± 0.022 | 0.832 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: arcee_fusion | 1.344 ± 0.012 | 1.787 ± 0.013 | 0.831 ± 0.003 | 0.000 ± 0.000 | 5 |

## N = 4

| | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 13.130 | 13.977 | 0.596 | 0.000 | – |
| **Full data** (one model) | 1.306 ± 0.011 | 1.723 ± 0.016 | 0.844 ± 0.001 | 0.000 ± 0.000 | 5 |
| Experts (each on 1/4 of the data) | 1.435 ± 0.007 | 1.908 ± 0.009 | 0.812 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: linear | 1.414 ± 0.007 | 1.887 ± 0.011 | 0.818 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: ties | 1.363 ± 0.006 | 1.811 ± 0.014 | 0.820 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: dare_ties | 1.421 ± 0.012 | 1.897 ± 0.010 | 0.816 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: task_arithmetic | 1.425 ± 0.005 | 1.901 ± 0.005 | 0.815 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: model_stock | 1.540 ± 0.019 | 2.032 ± 0.018 | 0.807 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: slerp | 1.423 ± 0.006 | 1.897 ± 0.009 | 0.816 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: arcee_fusion | 1.424 ± 0.014 | 1.895 ± 0.018 | 0.814 ± 0.001 | 0.000 ± 0.000 | 5 |

## N = 8

| | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 13.130 | 13.977 | 0.596 | 0.000 | – |
| **Full data** (one model) | 1.306 ± 0.011 | 1.723 ± 0.016 | 0.844 ± 0.001 | 0.000 ± 0.000 | 5 |
| Experts (each on 1/8 of the data) | 1.540 ± 0.008 | 2.036 ± 0.013 | 0.787 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: linear | 1.508 ± 0.010 | 2.006 ± 0.005 | 0.794 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: ties | 1.430 ± 0.012 | 1.891 ± 0.012 | 0.801 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: dare_ties | 1.512 ± 0.013 | 2.009 ± 0.009 | 0.794 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: task_arithmetic | 1.509 ± 0.015 | 2.006 ± 0.013 | 0.793 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: model_stock | 1.548 ± 0.019 | 2.047 ± 0.011 | 0.792 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: slerp | 1.512 ± 0.014 | 2.011 ± 0.012 | 0.792 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: arcee_fusion | 1.516 ± 0.009 | 2.000 ± 0.012 | 0.792 ± 0.005 | 0.000 ± 0.000 | 5 |
