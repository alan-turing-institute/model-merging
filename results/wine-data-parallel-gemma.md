# Data-parallel merging: Wine review points (80-100), Gemma 3 4B

Generated 2026-11-05 by `evaluate/make_data_parallel_tables.py` from the `.eval` logs in `evaluate/logs/`.

The training data is split into N equal parts; an expert is trained on each part (LoRA, from google/gemma-3-4b-it) and the N experts are merged. Every arm is repeated with seeds 1-5 (training data order and initialisation, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once. Test set: the 1,000 test reviews; mean absolute and root-mean-square error in points and Pearson correlation over the answered reviews, and the share with no number in the answer.

Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.

## N = 2

| | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 2.399 | 3.358 | 0.669 | 0.000 | – |
| **Full data** (one model) | 1.215 ± 0.005 | 1.599 ± 0.009 | 0.865 ± 0.002 | 0.000 ± 0.000 | 5 |
| Experts (each on 1/2 of the data) | 1.259 ± 0.011 | 1.655 ± 0.013 | 0.854 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: linear | 1.243 ± 0.024 | 1.638 ± 0.024 | 0.858 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: ties | 1.248 ± 0.013 | 1.650 ± 0.014 | 0.859 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: dare_ties | 1.241 ± 0.026 | 1.631 ± 0.027 | 0.859 ± 0.005 | 0.000 ± 0.000 | 5 |
| merge: task_arithmetic | 1.246 ± 0.016 | 1.640 ± 0.016 | 0.858 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: model_stock | 1.413 ± 0.025 | 1.868 ± 0.030 | 0.828 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: slerp | 1.244 ± 0.020 | 1.639 ± 0.022 | 0.858 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: arcee_fusion | 1.243 ± 0.018 | 1.638 ± 0.021 | 0.858 ± 0.004 | 0.000 ± 0.000 | 5 |

## N = 4

| | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 2.399 | 3.358 | 0.669 | 0.000 | – |
| **Full data** (one model) | 1.215 ± 0.005 | 1.599 ± 0.009 | 0.865 ± 0.002 | 0.000 ± 0.000 | 5 |
| Experts (each on 1/4 of the data) | 1.294 ± 0.005 | 1.699 ± 0.008 | 0.845 ± 0.001 | 0.000 ± 0.000 | 5 |
| merge: linear | 1.275 ± 0.008 | 1.670 ± 0.012 | 0.851 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: ties | 1.287 ± 0.012 | 1.681 ± 0.013 | 0.854 ± 0.001 | 0.000 ± 0.000 | 5 |
| merge: dare_ties | 1.277 ± 0.004 | 1.675 ± 0.007 | 0.851 ± 0.001 | 0.000 ± 0.000 | 5 |
| merge: task_arithmetic | 1.271 ± 0.006 | 1.664 ± 0.006 | 0.852 ± 0.001 | 0.000 ± 0.000 | 5 |
| merge: model_stock | 1.331 ± 0.013 | 1.762 ± 0.011 | 0.839 ± 0.001 | 0.000 ± 0.000 | 5 |
| merge: slerp | 1.275 ± 0.005 | 1.670 ± 0.011 | 0.851 ± 0.001 | 0.000 ± 0.000 | 5 |
| merge: arcee_fusion | 1.267 ± 0.019 | 1.661 ± 0.016 | 0.852 ± 0.001 | 0.000 ± 0.000 | 5 |

## N = 8

| | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 2.399 | 3.358 | 0.669 | 0.000 | – |
| **Full data** (one model) | 1.215 ± 0.005 | 1.599 ± 0.009 | 0.865 ± 0.002 | 0.000 ± 0.000 | 5 |
| Experts (each on 1/8 of the data) | 1.348 ± 0.005 | 1.772 ± 0.008 | 0.831 ± 0.001 | 0.000 ± 0.000 | 5 |
| merge: linear | 1.305 ± 0.007 | 1.727 ± 0.014 | 0.839 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: ties | 1.328 ± 0.013 | 1.719 ± 0.019 | 0.845 ± 0.004 | 0.000 ± 0.000 | 5 |
| merge: dare_ties | 1.305 ± 0.014 | 1.725 ± 0.016 | 0.840 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: task_arithmetic | 1.304 ± 0.012 | 1.730 ± 0.012 | 0.839 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: model_stock | 1.333 ± 0.008 | 1.766 ± 0.014 | 0.834 ± 0.003 | 0.000 ± 0.000 | 5 |
| merge: slerp | 1.305 ± 0.009 | 1.728 ± 0.013 | 0.839 ± 0.002 | 0.000 ± 0.000 | 5 |
| merge: arcee_fusion | 1.307 ± 0.009 | 1.721 ± 0.013 | 0.839 ± 0.002 | 0.000 ± 0.000 | 5 |
