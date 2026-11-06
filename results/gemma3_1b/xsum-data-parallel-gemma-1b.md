# Data-parallel merging: XSum summarisation (gold targets, long prompt), Gemma 3 1B

Generated 2026-11-05 by `evaluate/make_data_parallel_tables.py` from the `.eval` logs in `evaluate/logs-1b/`.

The training data is split into N equal parts; an expert is trained on each part (self-distillation, from google/gemma-3-1b-it) and the N experts are merged. Every arm is repeated with seeds 1-5 (training data order and initialisation, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once. Test set: 1,000 test articles, long prompt; ROUGE F-measures, and mean summary length in words.

Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.

## N = 2

| | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.223 | 0.040 | 0.166 | 19.6 | – |
| **Full data** (one model) | 0.303 ± 0.002 | 0.088 ± 0.002 | 0.230 ± 0.002 | 19.6 ± 0.5 | 5 |
| Experts (each on 1/2 of the data) | 0.301 ± 0.001 | 0.088 ± 0.001 | 0.229 ± 0.001 | 19.6 ± 0.2 | 5 |
| merge: linear | 0.303 ± 0.002 | 0.089 ± 0.001 | 0.230 ± 0.001 | 19.5 ± 0.2 | 5 |
| merge: ties | 0.304 ± 0.002 | 0.089 ± 0.001 | 0.232 ± 0.001 | 19.3 ± 0.1 | 5 |
| merge: dare_ties | 0.303 ± 0.002 | 0.089 ± 0.001 | 0.230 ± 0.001 | 19.6 ± 0.3 | 5 |
| merge: task_arithmetic | 0.302 ± 0.002 | 0.089 ± 0.001 | 0.230 ± 0.001 | 19.6 ± 0.2 | 5 |
| merge: model_stock | 0.297 ± 0.002 | 0.083 ± 0.001 | 0.225 ± 0.002 | 20.8 ± 0.2 | 5 |
| merge: slerp | 0.302 ± 0.002 | 0.089 ± 0.001 | 0.230 ± 0.001 | 19.6 ± 0.2 | 5 |
| merge: arcee_fusion | 0.303 ± 0.001 | 0.089 ± 0.001 | 0.230 ± 0.001 | 19.6 ± 0.2 | 5 |

## N = 4

| | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.223 | 0.040 | 0.166 | 19.6 | – |
| **Full data** (one model) | 0.303 ± 0.002 | 0.088 ± 0.002 | 0.230 ± 0.002 | 19.6 ± 0.5 | 5 |
| Experts (each on 1/4 of the data) | 0.296 ± 0.002 | 0.086 ± 0.001 | 0.227 ± 0.001 | 18.8 ± 0.1 | 5 |
| merge: linear | 0.297 ± 0.003 | 0.088 ± 0.001 | 0.228 ± 0.001 | 18.9 ± 0.2 | 5 |
| merge: ties | 0.299 ± 0.002 | 0.090 ± 0.001 | 0.231 ± 0.002 | 18.3 ± 0.2 | 5 |
| merge: dare_ties | 0.296 ± 0.003 | 0.087 ± 0.001 | 0.228 ± 0.002 | 18.8 ± 0.3 | 5 |
| merge: task_arithmetic | 0.297 ± 0.002 | 0.088 ± 0.001 | 0.228 ± 0.001 | 18.8 ± 0.3 | 5 |
| merge: model_stock | 0.295 ± 0.002 | 0.084 ± 0.001 | 0.225 ± 0.001 | 19.5 ± 0.2 | 5 |
| merge: slerp | 0.296 ± 0.002 | 0.087 ± 0.001 | 0.228 ± 0.001 | 18.9 ± 0.3 | 5 |
| merge: arcee_fusion | 0.297 ± 0.002 | 0.088 ± 0.001 | 0.229 ± 0.001 | 18.8 ± 0.2 | 5 |

## N = 8

| | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.223 | 0.040 | 0.166 | 19.6 | – |
| **Full data** (one model) | 0.303 ± 0.002 | 0.088 ± 0.002 | 0.230 ± 0.002 | 19.6 ± 0.5 | 5 |
| Experts (each on 1/8 of the data) | 0.286 ± 0.001 | 0.079 ± 0.001 | 0.218 ± 0.001 | 18.9 ± 0.2 | 5 |
| merge: linear | 0.285 ± 0.001 | 0.079 ± 0.001 | 0.218 ± 0.001 | 18.8 ± 0.2 | 5 |
| merge: ties | 0.288 ± 0.000 | 0.080 ± 0.001 | 0.221 ± 0.001 | 18.9 ± 0.2 | 5 |
| merge: dare_ties | 0.285 ± 0.001 | 0.080 ± 0.001 | 0.218 ± 0.001 | 18.8 ± 0.3 | 5 |
| merge: task_arithmetic | 0.284 ± 0.001 | 0.079 ± 0.001 | 0.217 ± 0.001 | 18.8 ± 0.3 | 5 |
| merge: model_stock | 0.284 ± 0.002 | 0.078 ± 0.001 | 0.217 ± 0.001 | 18.9 ± 0.3 | 5 |
| merge: slerp | 0.285 ± 0.002 | 0.079 ± 0.001 | 0.218 ± 0.001 | 18.9 ± 0.2 | 5 |
| merge: arcee_fusion | 0.284 ± 0.001 | 0.079 ± 0.001 | 0.218 ± 0.001 | 18.8 ± 0.3 | 5 |
