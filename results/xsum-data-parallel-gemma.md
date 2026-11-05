# Data-parallel merging: XSum summarisation (gold targets, long prompt), Gemma 3 4B

Generated 2026-11-05 by `evaluate/make_data_parallel_tables.py` from the `.eval` logs in `evaluate/logs/`.

The training data is split into N equal parts; an expert is trained on each part (LoRA for crime, self-distillation for XSum) and the N experts are merged. Every arm is repeated with seeds 1-5 (training data order and initialisation, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once. Test set: 1,000 test articles, long prompt; ROUGE F-measures, and mean summary length in words.

Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.

## N = 2

| | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.247 | 0.055 | 0.181 | 19.6 | – |
| **Full data** (one model) | 0.360 ± 0.007 | 0.135 ± 0.005 | 0.284 ± 0.006 | 18.2 ± 1.0 | 5 |
| Experts (each on 1/2 of the data) | 0.358 ± 0.002 | 0.132 ± 0.002 | 0.281 ± 0.003 | 19.3 ± 0.2 | 5 |
| merge: linear | 0.359 ± 0.002 | 0.132 ± 0.002 | 0.281 ± 0.002 | 19.3 ± 0.3 | 5 |
| merge: ties | 0.360 ± 0.001 | 0.134 ± 0.002 | 0.283 ± 0.001 | 18.7 ± 0.3 | 5 |
| merge: dare_ties | 0.358 ± 0.001 | 0.133 ± 0.002 | 0.281 ± 0.002 | 19.3 ± 0.3 | 5 |
| merge: task_arithmetic | 0.359 ± 0.002 | 0.132 ± 0.001 | 0.281 ± 0.001 | 19.3 ± 0.2 | 5 |
| merge: model_stock | 0.344 ± 0.001 | 0.118 ± 0.001 | 0.264 ± 0.001 | 21.6 ± 0.2 | 5 |
| merge: slerp | 0.358 ± 0.002 | 0.132 ± 0.002 | 0.281 ± 0.002 | 19.3 ± 0.3 | 5 |
| merge: arcee_fusion | 0.358 ± 0.002 | 0.132 ± 0.002 | 0.281 ± 0.002 | 19.6 ± 0.2 | 5 |

## N = 4

| | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.247 | 0.055 | 0.181 | 19.6 | – |
| **Full data** (one model) | 0.360 ± 0.007 | 0.135 ± 0.005 | 0.284 ± 0.006 | 18.2 ± 1.0 | 5 |
| Experts (each on 1/4 of the data) | 0.354 ± 0.002 | 0.130 ± 0.001 | 0.279 ± 0.001 | 19.6 ± 0.2 | 5 |
| merge: linear | 0.355 ± 0.003 | 0.131 ± 0.002 | 0.280 ± 0.002 | 19.6 ± 0.2 | 5 |
| merge: ties | 0.355 ± 0.003 | 0.133 ± 0.001 | 0.283 ± 0.001 | 18.7 ± 0.2 | 5 |
| merge: dare_ties | 0.355 ± 0.002 | 0.131 ± 0.001 | 0.280 ± 0.002 | 19.5 ± 0.2 | 5 |
| merge: task_arithmetic | 0.355 ± 0.002 | 0.132 ± 0.002 | 0.280 ± 0.002 | 19.6 ± 0.2 | 5 |
| merge: model_stock | 0.348 ± 0.001 | 0.124 ± 0.001 | 0.272 ± 0.001 | 21.0 ± 0.2 | 5 |
| merge: slerp | 0.355 ± 0.002 | 0.131 ± 0.002 | 0.280 ± 0.002 | 19.6 ± 0.2 | 5 |
| merge: arcee_fusion | 0.353 ± 0.002 | 0.129 ± 0.001 | 0.279 ± 0.001 | 19.6 ± 0.2 | 5 |

## N = 8

| | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.247 | 0.055 | 0.181 | 19.6 | – |
| **Full data** (one model) | 0.360 ± 0.007 | 0.135 ± 0.005 | 0.284 ± 0.006 | 18.2 ± 1.0 | 5 |
| Experts (each on 1/8 of the data) | 0.348 ± 0.000 | 0.125 ± 0.000 | 0.273 ± 0.000 | 19.6 ± 0.1 | 5 |
| merge: linear | 0.348 ± 0.001 | 0.125 ± 0.001 | 0.273 ± 0.001 | 19.7 ± 0.1 | 5 |
| merge: ties | 0.348 ± 0.002 | 0.127 ± 0.001 | 0.276 ± 0.002 | 18.3 ± 0.2 | 5 |
| merge: dare_ties | 0.348 ± 0.001 | 0.125 ± 0.001 | 0.273 ± 0.001 | 19.7 ± 0.1 | 5 |
| merge: task_arithmetic | 0.347 ± 0.000 | 0.125 ± 0.001 | 0.272 ± 0.001 | 19.7 ± 0.2 | 5 |
| merge: model_stock | 0.346 ± 0.001 | 0.123 ± 0.001 | 0.270 ± 0.001 | 20.2 ± 0.2 | 5 |
| merge: slerp | 0.348 ± 0.001 | 0.125 ± 0.000 | 0.273 ± 0.001 | 19.5 ± 0.1 | 5 |
| merge: arcee_fusion | 0.349 ± 0.001 | 0.126 ± 0.001 | 0.274 ± 0.001 | 19.9 ± 0.2 | 5 |
