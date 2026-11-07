# Merging results by median and interquartile range

Generated 2026-11-07 by `evaluate/make_merging_report.py` from the per-model JSONs in `results/` (as of commit 866ffd7).

Contents: data-parallel merging of crime and XSum (Gemma 3 4B); task merging (Gemma 3 4B, Mistral 7B, Llama 3.1 8B); whether the summary statistic changes the verdict; outlying seeds; an appendix with every task-merging metric.

Each cell is the median over seeds 1-5 with the interquartile range in brackets, *median (IQR)*, as each table's corner says; a cell over fewer seeds says how many; the base model is evaluated once. Quartiles interpolate linearly between the sorted values (Python's `statistics.quantiles` with `method="inclusive"`, numpy's default), so with five seeds Q1 and Q3 are the second and fourth smallest values and the IQR, Q3 - Q1, is the full width of the middle three seeds, not a ± half-width. For five seeds that scatter normally it averages 0.99 standard deviations, about 1.05 times the sample SD the mean ± SD reports give. That is only 0.73 of the IQR of the distribution the seeds come from (1.35 standard deviations), so these IQRs understate the spread, by the same factor in every five-seed cell, which leaves comparisons between arms unaffected. Q1 and Q3 are single seeds, so the IQR is itself noisy, and (0.000) means only that the second and fourth seeds agree at the precision shown. With three seeds the quartiles fall midway between neighbouring seeds and with two a quarter of the way in, so the IQR is half the range: one failing seed then moves it without limit, with two seeds the median is the mean, and such an IQR is not comparable with a five-seed cell's.

One seed that fails in a five-seed cell can move the median and IQR only as far as a neighbouring seed's value, whereas it can move the mean and standard deviation without limit. The price is that the median and IQR ignore the best and worst seed, so a method that fails on one seed in five can look reliable: cells with such a seed are marked † and listed under *Outlying seeds*.

The data-parallel tables are also in `results/crime-data-parallel-gemma.md` and `results/xsum-data-parallel-gemma.md` as mean ± SD. Those are built from the `.eval` logs, and these JSONs reproduce every row of them (`--stat mean`), so any difference comes from the statistic alone. Task merging's mean ± SD report is `results/task-merging.md`, whose *Retention* is R over all four tasks below, but it was made before the current task-merging results; `make_merging_report.py --stat mean` writes this report that way.

Every arm is repeated with seeds 1-5 (task merging is part-way through a re-run: see its section). The seed sets the training data order (for the LoRA experts also the adapter initialisation and dropout) and mergekit's random seed, which dare_ties uses to choose the weights it drops and arcee_fusion to sample the tensors its thresholds are estimated from.

Merge settings (`merge/make_merge_configs.py`, `merge/make_task_merge_configs.py`): linear, ties, dare_ties and task_arithmetic give each of the N models merged weight 1/N (N = 2, 4 or 8 in data-parallel merging, 4 in task merging); ties density 0.5; dare_ties density 0.3; model_stock takes no weights. slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges (for task merging, (crime + wine) + (CLINC150 + XSum)), with the first model of each pair as mergekit's base model; arcee_fusion keeps that base and adds the other model's selected changes, so it is not symmetric. With weights summing to 1 on a shared base, linear and task_arithmetic are the same model in exact arithmetic, so any difference between their rows is bf16 rounding and its effect on greedy decoding.

## Data-parallel merging: Crime classification (Reddit posts), Gemma 3 4B

The training data is split into N equal parts; an expert, a LoRA adapter, is trained on each part and the N experts are merged. Test set: the 1,077 test posts; accuracy and F1 for the crime class; the main metric is accuracy. *Experts* is the median over the N experts of each seed, then the median over seeds.

### Crime, N = 2

| median (IQR) | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.755 | 0.624 | – |
| **Full data** (one model) | 0.981 (0.002) | 0.979 (0.002) | 5 |
| Experts (each on 1/2 of the data) | 0.960 (0.005) | 0.954 (0.005) | 5 |
| merge: linear | 0.964 (0.005) | 0.959 (0.006) | 5 |
| merge: ties | 0.968 (0.006) | 0.963 (0.008) | 5 |
| merge: dare_ties | 0.965 (0.006) | 0.960 (0.008) | 5 |
| merge: task_arithmetic | 0.964 (0.003) | 0.959 (0.003) | 5 |
| merge: model_stock | 0.907 (0.015) | 0.885 (0.022) | 5 |
| merge: slerp | 0.964 (0.002) | 0.959 (0.002) | 5 |
| merge: arcee_fusion | 0.963 (0.006) | 0.958 (0.006) | 5 |

### Crime, N = 4

| median (IQR) | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.755 | 0.624 | – |
| **Full data** (one model) | 0.981 (0.002) | 0.979 (0.002) | 5 |
| Experts (each on 1/4 of the data) | 0.939 (0.001) | 0.930 (0.001) | 5 |
| merge: linear | 0.945 (0.011) | 0.938 (0.014) | 5 |
| merge: ties | 0.956 (0.003) | 0.951 (0.003) | 5 |
| merge: dare_ties | 0.946 (0.008) | 0.939 (0.010) | 5 |
| merge: task_arithmetic | 0.946 (0.011) | 0.939 (0.014) | 5 |
| merge: model_stock | 0.922 (0.007) | 0.905 (0.010) | 5 |
| merge: slerp | 0.946 (0.010) | 0.939 (0.012) | 5 |
| merge: arcee_fusion | 0.943 (0.009) | 0.936 (0.011) | 5 |

### Crime, N = 8

| median (IQR) | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.755 | 0.624 | – |
| **Full data** (one model) | 0.981 (0.002) | 0.979 (0.002) | 5 |
| Experts (each on 1/8 of the data) | 0.918 (0.003) | 0.907 (0.005) | 5 |
| merge: linear | 0.918 (0.004) | 0.903 (0.002) | 5 |
| merge: ties | 0.935 (0.006) | 0.926 (0.005) | 5 |
| merge: dare_ties | 0.918 (0.003) | 0.904 (0.004) | 5 |
| merge: task_arithmetic | 0.917 (0.003) | 0.903 (0.001) | 5 |
| merge: model_stock | 0.905 (0.005) | 0.886 (0.006) | 5 |
| merge: slerp | 0.918 (0.002) | 0.904 (0.001) | 5 |
| merge: arcee_fusion | 0.920 (0.008) | 0.907 (0.011) | 5 |

## Data-parallel merging: XSum summarisation (gold targets, long prompt), Gemma 3 4B

The training data is split into N equal parts; an expert, a student, by self-distillation, is trained on each part and the N experts are merged. Test set: 1,000 test articles, long prompt; ROUGE F-measures, and mean summary length in words; the main metric is ROUGE-1. *Experts* is the median over the N experts of each seed, then the median over seeds.

### XSum, N = 2

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.247 | 0.055 | 0.181 | 19.6 | – |
| **Full data** (one model) | 0.360 (0.006) | 0.133 (0.006) | 0.283 (0.003) | 18.2 (0.9) | 5 |
| Experts (each on 1/2 of the data) | 0.359 (0.001) † | 0.132 (0.002) | 0.281 (0.003) | 19.4 (0.4) | 5 |
| merge: linear | 0.359 (0.003) | 0.133 (0.002) | 0.281 (0.003) | 19.4 (0.4) | 5 |
| merge: ties | 0.360 (0.002) | 0.134 (0.002) | 0.283 (0.003) | 18.8 (0.3) | 5 |
| merge: dare_ties | 0.358 (0.001) | 0.133 (0.002) | 0.281 (0.000) | 19.3 (0.5) | 5 |
| merge: task_arithmetic | 0.359 (0.001) | 0.133 (0.002) | 0.282 (0.002) | 19.3 (0.3) | 5 |
| merge: model_stock | 0.344 (0.001) | 0.118 (0.002) | 0.264 (0.002) | 21.6 (0.3) | 5 |
| merge: slerp | 0.358 (0.002) | 0.133 (0.003) | 0.281 (0.002) | 19.3 (0.4) | 5 |
| merge: arcee_fusion | 0.359 (0.003) | 0.132 (0.002) | 0.281 (0.004) | 19.6 (0.2) | 5 |

### XSum, N = 4

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.247 | 0.055 | 0.181 | 19.6 | – |
| **Full data** (one model) | 0.360 (0.006) | 0.133 (0.006) | 0.283 (0.003) | 18.2 (0.9) | 5 |
| Experts (each on 1/4 of the data) | 0.353 (0.001) | 0.131 (0.001) | 0.279 (0.000) | 19.6 (0.1) | 5 |
| merge: linear | 0.357 (0.003) | 0.131 (0.002) | 0.280 (0.001) | 19.6 (0.1) | 5 |
| merge: ties | 0.355 (0.004) | 0.132 (0.002) | 0.282 (0.002) | 18.7 (0.3) | 5 |
| merge: dare_ties | 0.355 (0.002) | 0.132 (0.002) | 0.281 (0.001) | 19.5 (0.1) | 5 |
| merge: task_arithmetic | 0.356 (0.001) | 0.132 (0.002) | 0.280 (0.001) | 19.6 (0.2) | 5 |
| merge: model_stock | 0.349 (0.002) | 0.124 (0.001) | 0.272 (0.001) | 20.9 (0.3) | 5 |
| merge: slerp | 0.355 (0.000) † | 0.132 (0.001) | 0.281 (0.002) | 19.4 (0.2) | 5 |
| merge: arcee_fusion | 0.353 (0.001) | 0.129 (0.001) | 0.279 (0.001) | 19.6 (0.1) | 5 |

### XSum, N = 8

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.247 | 0.055 | 0.181 | 19.6 | – |
| **Full data** (one model) | 0.360 (0.006) | 0.133 (0.006) | 0.283 (0.003) | 18.2 (0.9) | 5 |
| Experts (each on 1/8 of the data) | 0.347 (0.000) | 0.125 (0.000) | 0.273 (0.001) | 19.7 (0.2) | 5 |
| merge: linear | 0.348 (0.001) | 0.125 (0.001) | 0.272 (0.002) | 19.7 (0.2) | 5 |
| merge: ties | 0.348 (0.003) | 0.127 (0.002) | 0.276 (0.003) | 18.3 (0.1) | 5 |
| merge: dare_ties | 0.348 (0.001) | 0.125 (0.001) | 0.274 (0.001) | 19.7 (0.3) | 5 |
| merge: task_arithmetic | 0.348 (0.000) | 0.125 (0.001) | 0.273 (0.002) | 19.7 (0.3) | 5 |
| merge: model_stock | 0.346 (0.001) | 0.122 (0.000) | 0.270 (0.001) | 20.3 (0.2) | 5 |
| merge: slerp | 0.348 (0.000) | 0.125 (0.000) | 0.273 (0.001) | 19.5 (0.1) | 5 |
| merge: arcee_fusion | 0.349 (0.003) | 0.126 (0.000) | 0.275 (0.001) | 19.9 (0.2) | 5 |

## Task merging

Four experts, one per task, each trained on all of its task's data (MLP-only: LoRA for crime, wine and CLINC150; self-distillation for XSum), are merged into one model, which is evaluated on all four tasks. Each method merges the same four experts. The MoE (Mistral and Llama only: mergekit has no MoE path for Gemma) routes each token, at every layer, to one expert's MLP; its attention, embeddings and norms are the base model's. The prompt_mean MoE is the same MoE with each layer's router set from the router's own inputs on example prompts in the evaluation's format (`merge/set_moe_router.py`), instead of mergekit's hidden-state router. The MoE with a random router routes without task information, a control for how much the routing matters; it is left out of the verdict. Test sets: Crime, 1,077 test posts; Wine, 1,000 test reviews (error in points: lower is better); CLINC150, 4,500 test queries, labels prompt; XSum, 1,000 test articles, long prompt.

Task merging was re-run after XSum's self-distillation targets were regenerated as 18-24 word summaries (866ffd7): the re-run merges replaced the earlier ones, and the earlier results not yet re-run were moved to `results/superseded/`. The experts have seeds 1-5, in every model; the merges so far: seeds 1 and 2, with part of seed 3, in every model. Cells over fewer than five seeds say how many, and their IQR is not comparable with a five-seed cell's (see the start).

*Recovery R* is the share of the expert's gain over the base model that a merge keeps, for one task and seed: (merge - base) / (expert - base), with the expert of the same seed, on the task's main metric (crime accuracy, wine MAE, CLINC150 accuracy, XSum ROUGE-1: the columns of each model's first table). 1 is as good as the expert, 0 no better than the untrained model, and below 0 worse than it. *All four* is the mean of a seed's four R values, then the median over seeds; it is a mean because it is a score, and a median of four would ignore the best and the worst task.

### Gemma 3 4B (`google/gemma-3-4b-it`)

| median (IQR) | Crime Accuracy | Wine MAE | CLINC150 Accuracy | XSum ROUGE-1 |
|---|---|---|---|---|
| Base model (untrained) | 0.755 | 2.376 | 0.704 | 0.247 |
| Expert (trained on that task) | 0.980 (0.002) | 1.179 (0.005) | 0.978 (0.000) | 0.361 (0.007) † |
| merge: linear | 0.894 (0.001, 3 seeds) | 1.403 (0.042, 3 seeds) | 0.962 (0.003, 3 seeds) | 0.297 (0.003, 3 seeds) |
| merge: ties | 0.941 (0.011, 3 seeds) | 1.236 (0.978, 3 seeds) | 0.977 (0.001, 3 seeds) | 0.246 (0.019, 3 seeds) |
| merge: dare_ties | 0.894 (0.002, 3 seeds) | 1.400 (0.027, 3 seeds) | 0.964 (0.002, 3 seeds) | 0.295 (0.004, 3 seeds) |
| merge: task_arithmetic | 0.894 (0.002, 3 seeds) | 1.406 (0.039, 3 seeds) | 0.962 (0.001, 2 seeds) | 0.298 (0.002, 3 seeds) |
| merge: model_stock | 0.761 (0.002, 3 seeds) | 2.379 (0.012, 3 seeds) | 0.717 (0.000, 2 seeds) | 0.247 (0.001, 3 seeds) |
| merge: slerp | 0.894 (0.001, 3 seeds) | 1.403 (0.040, 3 seeds) | 0.962 (0.000, 2 seeds) | 0.299 (0.002, 3 seeds) |
| merge: arcee_fusion | 0.929 (0.006, 3 seeds) | 1.730 (0.047, 3 seeds) | 0.936 (0.004, 3 seeds) | 0.267 (0.004, 3 seeds) |

Recovery R (median over seeds):

| median (IQR) | Crime | Wine | CLINC150 | XSum | All four |
|---|---|---|---|---|---|
| merge: linear | 0.62 (0.01, 3 seeds) | 0.80 (0.03, 3 seeds) | 0.94 (0.01, 3 seeds) | 0.47 (0.09, 3 seeds) | 0.70 (0.02, 3 seeds) |
| merge: ties | 0.83 (0.04, 3 seeds) | 0.95 (0.81, 3 seeds) | 1.00 (0.00, 3 seeds) | -0.01 (0.16, 3 seeds) | 0.69 (0.23, 3 seeds) |
| merge: dare_ties | 0.63 (0.01, 3 seeds) | 0.80 (0.02, 3 seeds) | 0.95 (0.01, 3 seeds) | 0.46 (0.08, 3 seeds) | 0.71 (0.02, 3 seeds) |
| merge: task_arithmetic | 0.63 (0.01, 3 seeds) | 0.80 (0.03, 3 seeds) | 0.94 (0.00, 2 seeds) | 0.47 (0.10, 3 seeds) | 0.72 (0.03, 2 seeds) |
| merge: model_stock | 0.03 (0.01, 3 seeds) | 0.00 (0.01, 3 seeds) | 0.05 (0.00, 2 seeds) | 0.00 (0.01, 3 seeds) | 0.02 (0.00, 2 seeds) |
| merge: slerp | 0.62 (0.00, 3 seeds) | 0.81 (0.03, 3 seeds) | 0.94 (0.00, 2 seeds) | 0.48 (0.10, 3 seeds) | 0.73 (0.03, 2 seeds) |
| merge: arcee_fusion | 0.78 (0.03, 3 seeds) | 0.54 (0.03, 3 seeds) | 0.85 (0.02, 3 seeds) | 0.22 (0.05, 3 seeds) | 0.60 (0.02, 3 seeds) |

### Mistral 7B (`mistralai/Mistral-7B-Instruct-v0.3`)

| median (IQR) | Crime Accuracy | Wine MAE | CLINC150 Accuracy | XSum ROUGE-1 |
|---|---|---|---|---|
| Base model (untrained) | 0.774 | 1.676 | 0.454 | 0.238 |
| Expert (trained on that task) | 0.983 (0.001) | 1.137 (0.016) | 0.975 (0.002) | 0.375 (0.004) |
| merge: linear | 0.927 (0.006, 2 seeds) | 1.264 (0.014, 2 seeds) | 0.972 (0.001, 2 seeds) | 0.325 (0.006, 2 seeds) |
| merge: ties | 0.946 (0.007, 2 seeds) | 1.184 (0.012, 2 seeds) | 0.976 (0.001, 2 seeds) | 0.059 (0.027, 2 seeds) |
| merge: dare_ties | 0.925 (0.005, 2 seeds) | 1.265 (0.012, 2 seeds) | 0.972 (0.001, 2 seeds) | 0.313 (0.006, 2 seeds) |
| merge: task_arithmetic | 0.927 (0.007, 2 seeds) | 1.259 (0.014, 3 seeds) | 0.973 (0.002, 3 seeds) | 0.329 (0.006, 3 seeds) |
| merge: model_stock | 0.780 (0.001, 3 seeds) | 1.648 (0.012, 3 seeds) | 0.472 (0.002, 3 seeds) | 0.237 (0.001, 3 seeds) |
| merge: slerp | 0.930 (0.008, 3 seeds) | 1.260 (0.014, 3 seeds) | 0.974 (0.003, 3 seeds) | 0.329 (0.005, 3 seeds) |
| merge: arcee_fusion | 0.979 (0.002, 3 seeds) | 1.331 (0.035, 3 seeds) | 0.963 (0.003, 3 seeds) | 0.172 (0.041, 3 seeds) |
| MoE (mergekit-moe) | 0.904 (0.020, 3 seeds) | 1.209 (0.016, 3 seeds) | 0.975 (0.001, 3 seeds) | 0.349 (0.016, 3 seeds) |
| MoE, prompt_mean router | 0.976 (0.000, 3 seeds) | 1.146 (0.006, 3 seeds) | 0.975 (0.002, 3 seeds) | 0.287 (0.027, 3 seeds) |
| MoE, random router (control) | 0.916 (0.008, 3 seeds) | 1.351 (0.031, 3 seeds) | 0.949 (0.000, 3 seeds) | 0.327 (0.031, 3 seeds) |

Recovery R (median over seeds):

| median (IQR) | Crime | Wine | CLINC150 | XSum | All four |
|---|---|---|---|---|---|
| merge: linear | 0.73 (0.03, 2 seeds) | 0.77 (0.03, 2 seeds) | 0.99 (0.00, 2 seeds) | 0.64 (0.04, 2 seeds) | 0.78 (0.02, 2 seeds) |
| merge: ties | 0.82 (0.04, 2 seeds) | 0.92 (0.03, 2 seeds) | 1.00 (0.00, 2 seeds) | -1.32 (0.22, 2 seeds) | 0.36 (0.07, 2 seeds) |
| merge: dare_ties | 0.72 (0.03, 2 seeds) | 0.77 (0.03, 2 seeds) | 0.99 (0.00, 2 seeds) | 0.55 (0.04, 2 seeds) | 0.76 (0.02, 2 seeds) |
| merge: task_arithmetic | 0.73 (0.04, 2 seeds) | 0.76 (0.03, 3 seeds) | 1.00 (0.00, 3 seeds) | 0.66 (0.03, 3 seeds) | 0.78 (0.02, 2 seeds) |
| merge: model_stock | 0.03 (0.01, 3 seeds) | 0.05 (0.02, 3 seeds) | 0.03 (0.00, 3 seeds) | 0.00 (0.01, 3 seeds) | 0.02 (0.00, 3 seeds) |
| merge: slerp | 0.75 (0.03, 3 seeds) | 0.76 (0.03, 3 seeds) | 1.00 (0.00, 3 seeds) | 0.67 (0.03, 3 seeds) | 0.80 (0.02, 3 seeds) |
| merge: arcee_fusion | 0.97 (0.01, 3 seeds) | 0.63 (0.07, 3 seeds) | 0.98 (0.00, 3 seeds) | -0.48 (0.31, 3 seeds) | 0.53 (0.09, 3 seeds) |
| MoE (mergekit-moe) | 0.62 (0.09, 3 seeds) | 0.88 (0.02, 3 seeds) | 1.00 (0.00, 3 seeds) | 0.82 (0.11, 3 seeds) | 0.82 (0.03, 3 seeds) |
| MoE, prompt_mean router | 0.96 (0.00, 3 seeds) | 0.98 (0.01, 3 seeds) | 1.00 (0.00, 3 seeds) | 0.37 (0.20, 3 seeds) | 0.83 (0.05, 3 seeds) |
| MoE, random router (control) | 0.68 (0.04, 3 seeds) | 0.61 (0.05, 3 seeds) | 0.95 (0.00, 3 seeds) | 0.65 (0.21, 3 seeds) | 0.76 (0.06, 3 seeds) |

### Llama 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`)

| median (IQR) | Crime Accuracy | Wine MAE | CLINC150 Accuracy | XSum ROUGE-1 |
|---|---|---|---|---|
| Base model (untrained) | 0.743 | 4.356 | 0.640 | 0.289 |
| Expert (trained on that task) | 0.985 (0.004) | 1.140 (0.016) | 0.979 (0.001) | 0.378 (0.001) |
| merge: linear | 0.898 (0.000, 2 seeds) | 1.251 (0.014, 3 seeds) | 0.965 (0.002, 2 seeds) | 0.338 (0.002, 2 seeds) |
| merge: ties | 0.945 (0.002, 2 seeds) | 1.187 (0.015, 2 seeds) | 0.979 (0.001, 3 seeds) | 0.260 (0.005, 2 seeds) |
| merge: dare_ties | 0.900 (0.002, 2 seeds) | 1.243 (0.001, 2 seeds) | 0.967 (0.003, 2 seeds) | 0.335 (0.001, 3 seeds) |
| merge: task_arithmetic | 0.898 (0.000, 2 seeds) | 1.252 (0.012, 3 seeds) | 0.965 (0.002, 2 seeds) | 0.338 (0.002, 3 seeds) |
| merge: model_stock | 0.751 (0.001, 2 seeds) | 4.166 (0.003, 2 seeds) | 0.651 (0.001, 2 seeds) | 0.289 (0.001, 2 seeds) |
| merge: slerp | 0.898 (0.003, 3 seeds) | 1.244 (0.003, 2 seeds) | 0.965 (0.002, 2 seeds) | 0.338 (0.002, 3 seeds) |
| merge: arcee_fusion | 0.967 (0.001, 2 seeds) | 1.348 (0.014, 2 seeds) | 0.949 (0.003, 3 seeds) | 0.312 (0.003, 3 seeds) |
| MoE (mergekit-moe) | 0.923 (0.007, 3 seeds) | 1.191 (0.021, 3 seeds) | 0.976 (0.001, 3 seeds) | 0.356 (0.001, 3 seeds) |
| MoE, prompt_mean router | 0.974 (0.009, 3 seeds) | 1.144 (0.008, 3 seeds) | 0.978 (0.001, 3 seeds) | 0.335 (0.005, 3 seeds) |
| MoE, random router (control) | 0.902 (0.006, 3 seeds) | 1.282 (0.062, 3 seeds) | 0.958 (0.012, 3 seeds) | 0.335 (0.006, 3 seeds) |

Recovery R (median over seeds):

| median (IQR) | Crime | Wine | CLINC150 | XSum | All four |
|---|---|---|---|---|---|
| merge: linear | 0.64 (0.00, 2 seeds) | 0.97 (0.01, 3 seeds) | 0.96 (0.01, 2 seeds) | 0.55 (0.02, 2 seeds) | 0.78 (0.01, 2 seeds) |
| merge: ties | 0.84 (0.00, 2 seeds) | 0.99 (0.01, 2 seeds) | 1.00 (0.00, 3 seeds) | -0.32 (0.06, 2 seeds) | 0.63 (0.01, 2 seeds) |
| merge: dare_ties | 0.65 (0.01, 2 seeds) | 0.98 (0.00, 2 seeds) | 0.96 (0.01, 2 seeds) | 0.51 (0.02, 3 seeds) | 0.77 (0.01, 2 seeds) |
| merge: task_arithmetic | 0.64 (0.00, 2 seeds) | 0.97 (0.01, 3 seeds) | 0.96 (0.01, 2 seeds) | 0.55 (0.02, 3 seeds) | 0.78 (0.01, 2 seeds) |
| merge: model_stock | 0.03 (0.01, 2 seeds) | 0.06 (0.00, 2 seeds) | 0.03 (0.00, 2 seeds) | 0.00 (0.01, 2 seeds) | 0.03 (0.00, 2 seeds) |
| merge: slerp | 0.65 (0.02, 3 seeds) | 0.98 (0.00, 2 seeds) | 0.96 (0.01, 2 seeds) | 0.54 (0.02, 3 seeds) | 0.78 (0.01, 2 seeds) |
| merge: arcee_fusion | 0.93 (0.01, 2 seeds) | 0.94 (0.01, 2 seeds) | 0.91 (0.01, 3 seeds) | 0.26 (0.03, 3 seeds) | 0.75 (0.01, 2 seeds) |
| MoE (mergekit-moe) | 0.75 (0.03, 3 seeds) | 0.99 (0.01, 3 seeds) | 0.99 (0.00, 3 seeds) | 0.75 (0.02, 3 seeds) | 0.88 (0.01, 3 seeds) |
| MoE, prompt_mean router | 0.97 (0.03, 3 seeds) | 1.00 (0.01, 3 seeds) | 0.99 (0.00, 3 seeds) | 0.52 (0.06, 3 seeds) | 0.88 (0.02, 3 seeds) |
| MoE, random router (control) | 0.67 (0.03, 3 seeds) | 0.97 (0.02, 3 seeds) | 0.93 (0.03, 3 seeds) | 0.51 (0.07, 3 seeds) | 0.77 (0.04, 3 seeds) |

## Does the summary statistic change the verdict?

In 9 of the 9 comparisons the best merge by the median is also the best by the mean, ties at the precision shown counting as best. model_stock is the worst merge, or tied worst, in every comparison under both statistics.

The main metric of each comparison is accuracy (crime), ROUGE-1 (XSum) and R over all four tasks (task merging). Methods are tied when they agree at the precision shown, and tied methods are listed together, the better unrounded first. *Next* is the methods with the second-best median. *Wins* counts the seeds on which the first best method beats each of them, out of the seeds both have: every method of a seed merges the same experts, so the comparison is paired. With five seeds or fewer neither the medians nor the win counts are a significance test; they show how consistent a lead is. *Same best* says whether the best by the median and by the mean have a method in common. Where the arms of a comparison have different seeds, its row uses only the seeds all of them have, named after it, so its medians can differ from the table's.

| Comparison | Best by median | Next by median | Wins | Best by mean | Same best |
|---|---|---|---|---|---|
| Data-parallel Crime, N = 2 | ties 0.968 | dare_ties 0.965 | 4 of 5 | ties 0.968 | yes |
| Data-parallel Crime, N = 4 | ties 0.956 | dare_ties = task_arithmetic = slerp 0.946 | 5 of 5, 5 of 5, 5 of 5 | ties 0.957 | yes |
| Data-parallel Crime, N = 8 | ties 0.935 | arcee_fusion 0.920 | 5 of 5 | ties 0.935 | yes |
| Data-parallel XSum, N = 2 | ties 0.360 | task_arithmetic = linear = arcee_fusion 0.359 | 3 of 5, 3 of 5, 4 of 5 | ties 0.360 | yes |
| Data-parallel XSum, N = 4 | linear 0.357 | task_arithmetic 0.356 | 4 of 5 | linear = task_arithmetic = ties = dare_ties = slerp 0.355 | yes |
| Data-parallel XSum, N = 8 | arcee_fusion 0.349 | ties = dare_ties = linear = slerp = task_arithmetic 0.348 | 3 of 5, 4 of 5, 5 of 5, 4 of 5, 5 of 5 | arcee_fusion 0.349 | yes |
| Task merging, Gemma 3 4B (seeds 1 and 2) | slerp 0.73 | task_arithmetic = dare_ties = linear 0.72 | 2 of 2, 1 of 2, 2 of 2 | slerp 0.73 | yes |
| Task merging, Mistral 7B (seeds 1 and 2) | MoE, prompt_mean router 0.82 | MoE (mergekit-moe) 0.80 | 1 of 2 | MoE, prompt_mean router 0.82 | yes |
| Task merging, Llama 3.1 8B (seeds 1 and 2) | MoE, prompt_mean router 0.89 | MoE (mergekit-moe) 0.88 | 2 of 2 | MoE, prompt_mean router 0.89 | yes |

## Outlying seeds

Cells in which a seed lies more than 3 IQRs beyond the nearer quartile, taking as the IQR the larger of the cell's own and the median IQR of that metric over every arm of the study (cells over fewer seeds included) (with five seeds a cell's own IQR is often a unit or two of the last digit, which would make ordinary noise outlying): 5 of the 180 distinct five-seed cells, the appendix included; cells over fewer seeds are not checked. They are marked † in the tables. The median and IQR barely move for such a seed, which is the point of them, but they also hide it, and a method that fails on one seed in five is less reliable than its median suggests. The outlying values are in bold; *Mean ± SD* is computed as in the mean ± SD reports.

| Where | Row | Metric | Seeds 1-5 | Mean ± SD | Median (IQR) |
|---|---|---|---|---|---|
| Data-parallel XSum, N = 2 | Experts (each on 1/2 of the data) | ROUGE-1 | **0.354**, 0.359, 0.359, 0.360, 0.358 | 0.358 ± 0.002 | 0.359 (0.001) |
| Data-parallel XSum, N = 4 | merge: slerp | ROUGE-1 | 0.356, **0.351**, 0.356, 0.355, 0.355 | 0.355 ± 0.002 | 0.355 (0.000) |
| Task merging, Gemma 3 4B | Expert (trained on that task) | XSum ROUGE-1 | **0.327**, 0.363, 0.361, 0.364, 0.356 | 0.354 ± 0.016 | 0.361 (0.007) |
| Task merging, Gemma 3 4B | Expert (trained on that task) | XSum ROUGE-2 | **0.112**, 0.134, 0.135, 0.138, 0.131 | 0.130 ± 0.010 | 0.134 (0.005) |
| Task merging, Gemma 3 4B | Expert (trained on that task) | XSum ROUGE-L | **0.258**, 0.283, 0.283, 0.287, 0.280 | 0.278 ± 0.012 | 0.283 (0.003) |

By arm and seed, with the arm's score on that seed and on its other four, on the main metric of its comparison (as in the verdict above; an expert on its own task's), so that a seed can be judged as a whole: a seed outlying on one metric may still be weak, or fine, overall.

| Where | Row | Seed | Outlying on | Score on that seed | Other seeds |
|---|---|---|---|---|---|
| Data-parallel XSum, N = 2 | Experts (each on 1/2 of the data) | 1 | ROUGE-1 | ROUGE-1 0.354 | 0.358 to 0.360 |
| Data-parallel XSum, N = 4 | merge: slerp | 2 | ROUGE-1 | ROUGE-1 0.351 | 0.355 to 0.356 |
| Task merging, Gemma 3 4B | Expert (trained on that task) | 1 | XSum ROUGE-1, XSum ROUGE-2, XSum ROUGE-L | XSum ROUGE-1 0.327 | 0.356 to 0.364 |

## Appendix: every metric, task merging

### Gemma 3 4B: Crime

| median (IQR) | Accuracy | F1 |
|---|---|---|
| Base model (untrained) | 0.755 | 0.624 |
| Expert (trained on that task) | 0.980 (0.002) | 0.977 (0.002) |
| merge: linear | 0.894 (0.001, 3 seeds) | 0.866 (0.002, 3 seeds) |
| merge: ties | 0.941 (0.011, 3 seeds) | 0.933 (0.014, 3 seeds) |
| merge: dare_ties | 0.894 (0.002, 3 seeds) | 0.867 (0.003, 3 seeds) |
| merge: task_arithmetic | 0.894 (0.002, 3 seeds) | 0.866 (0.002, 3 seeds) |
| merge: model_stock | 0.761 (0.002, 3 seeds) | 0.638 (0.004, 3 seeds) |
| merge: slerp | 0.894 (0.001, 3 seeds) | 0.867 (0.002, 3 seeds) |
| merge: arcee_fusion | 0.929 (0.006, 3 seeds) | 0.915 (0.008, 3 seeds) |

### Gemma 3 4B: Wine

| median (IQR) | MAE | RMSE | Pearson |
|---|---|---|---|
| Base model (untrained) | 2.376 | 3.322 | 0.671 |
| Expert (trained on that task) | 1.179 (0.005) | 1.573 (0.030) | 0.870 (0.007) |
| merge: linear | 1.403 (0.042, 3 seeds) | 1.878 (0.100, 3 seeds) | 0.824 (0.016, 3 seeds) |
| merge: ties | 1.236 (0.978, 3 seeds) | 1.626 (29.697, 3 seeds) | 0.858 (0.405, 3 seeds) |
| merge: dare_ties | 1.400 (0.027, 3 seeds) | 1.854 (0.077, 3 seeds) | 0.833 (0.013, 3 seeds) |
| merge: task_arithmetic | 1.406 (0.039, 3 seeds) | 1.880 (0.093, 3 seeds) | 0.823 (0.015, 3 seeds) |
| merge: model_stock | 2.379 (0.012, 3 seeds) | 3.319 (0.017, 3 seeds) | 0.672 (0.001, 3 seeds) |
| merge: slerp | 1.403 (0.040, 3 seeds) | 1.881 (0.091, 3 seeds) | 0.823 (0.014, 3 seeds) |
| merge: arcee_fusion | 1.730 (0.047, 3 seeds) | 2.493 (0.109, 3 seeds) | 0.739 (0.012, 3 seeds) |

### Gemma 3 4B: CLINC150

| median (IQR) | Accuracy |
|---|---|
| Base model (untrained) | 0.704 |
| Expert (trained on that task) | 0.978 (0.000) |
| merge: linear | 0.962 (0.003, 3 seeds) |
| merge: ties | 0.977 (0.001, 3 seeds) |
| merge: dare_ties | 0.964 (0.002, 3 seeds) |
| merge: task_arithmetic | 0.962 (0.001, 2 seeds) |
| merge: model_stock | 0.717 (0.000, 2 seeds) |
| merge: slerp | 0.962 (0.000, 2 seeds) |
| merge: arcee_fusion | 0.936 (0.004, 3 seeds) |

### Gemma 3 4B: XSum

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |
|---|---|---|---|---|
| Base model (untrained) | 0.247 | 0.055 | 0.181 | 19.6 |
| Expert (trained on that task) | 0.361 (0.007) † | 0.134 (0.005) † | 0.283 (0.003) † | 19.3 (0.6) |
| merge: linear | 0.297 (0.003, 3 seeds) | 0.087 (0.001, 3 seeds) | 0.228 (0.003, 3 seeds) | 18.8 (0.5, 3 seeds) |
| merge: ties | 0.246 (0.019, 3 seeds) | 0.058 (0.008, 3 seeds) | 0.192 (0.011, 3 seeds) | 14.1 (1.7, 3 seeds) |
| merge: dare_ties | 0.295 (0.004, 3 seeds) | 0.085 (0.002, 3 seeds) | 0.225 (0.003, 3 seeds) | 19.0 (0.6, 3 seeds) |
| merge: task_arithmetic | 0.298 (0.002, 3 seeds) | 0.088 (0.001, 3 seeds) | 0.228 (0.003, 3 seeds) | 18.9 (0.5, 3 seeds) |
| merge: model_stock | 0.247 (0.001, 3 seeds) | 0.055 (0.000, 3 seeds) | 0.183 (0.000, 3 seeds) | 19.6 (0.0, 3 seeds) |
| merge: slerp | 0.299 (0.002, 3 seeds) | 0.089 (0.001, 3 seeds) | 0.229 (0.003, 3 seeds) | 19.0 (0.5, 3 seeds) |
| merge: arcee_fusion | 0.267 (0.004, 3 seeds) | 0.067 (0.002, 3 seeds) | 0.202 (0.003, 3 seeds) | 19.0 (1.0, 3 seeds) |

### Mistral 7B: Crime

| median (IQR) | Accuracy | F1 |
|---|---|---|
| Base model (untrained) | 0.774 | 0.682 |
| Expert (trained on that task) | 0.983 (0.001) | 0.981 (0.001) |
| merge: linear | 0.927 (0.006, 2 seeds) | 0.911 (0.008, 2 seeds) |
| merge: ties | 0.946 (0.007, 2 seeds) | 0.937 (0.009, 2 seeds) |
| merge: dare_ties | 0.925 (0.005, 2 seeds) | 0.909 (0.007, 2 seeds) |
| merge: task_arithmetic | 0.927 (0.007, 2 seeds) | 0.911 (0.009, 2 seeds) |
| merge: model_stock | 0.780 (0.001, 3 seeds) | 0.696 (0.002, 3 seeds) |
| merge: slerp | 0.930 (0.008, 3 seeds) | 0.915 (0.010, 3 seeds) |
| merge: arcee_fusion | 0.979 (0.002, 3 seeds) | 0.976 (0.003, 3 seeds) |
| MoE (mergekit-moe) | 0.904 (0.020, 3 seeds) | 0.887 (0.020, 3 seeds) |
| MoE, prompt_mean router | 0.976 (0.000, 3 seeds) | 0.972 (0.000, 3 seeds) |
| MoE, random router (control) | 0.916 (0.008, 3 seeds) | 0.896 (0.011, 3 seeds) |

### Mistral 7B: Wine

| median (IQR) | MAE | RMSE | Pearson |
|---|---|---|---|
| Base model (untrained) | 1.676 | 2.255 | 0.740 |
| Expert (trained on that task) | 1.137 (0.016) | 1.511 (0.006) | 0.878 (0.002) |
| merge: linear | 1.264 (0.014, 2 seeds) | 1.675 (0.014, 2 seeds) | 0.865 (0.000, 2 seeds) |
| merge: ties | 1.184 (0.012, 2 seeds) | 1.567 (0.021, 2 seeds) | 0.870 (0.002, 2 seeds) |
| merge: dare_ties | 1.265 (0.012, 2 seeds) | 1.672 (0.009, 2 seeds) | 0.865 (0.001, 2 seeds) |
| merge: task_arithmetic | 1.259 (0.014, 3 seeds) | 1.664 (0.017, 3 seeds) | 0.865 (0.001, 3 seeds) |
| merge: model_stock | 1.648 (0.012, 3 seeds) | 2.186 (0.026, 3 seeds) | 0.753 (0.005, 3 seeds) |
| merge: slerp | 1.260 (0.014, 3 seeds) | 1.667 (0.013, 3 seeds) | 0.865 (0.001, 3 seeds) |
| merge: arcee_fusion | 1.331 (0.035, 3 seeds) | 1.743 (0.043, 3 seeds) | 0.847 (0.002, 3 seeds) |
| MoE (mergekit-moe) | 1.209 (0.016, 3 seeds) | 1.602 (0.019, 3 seeds) | 0.868 (0.000, 3 seeds) |
| MoE, prompt_mean router | 1.146 (0.006, 3 seeds) | 1.520 (0.008, 3 seeds) | 0.875 (0.002, 3 seeds) |
| MoE, random router (control) | 1.351 (0.031, 3 seeds) | 1.798 (0.037, 3 seeds) | 0.840 (0.006, 3 seeds) |

### Mistral 7B: CLINC150

| median (IQR) | Accuracy |
|---|---|
| Base model (untrained) | 0.454 |
| Expert (trained on that task) | 0.975 (0.002) |
| merge: linear | 0.972 (0.001, 2 seeds) |
| merge: ties | 0.976 (0.001, 2 seeds) |
| merge: dare_ties | 0.972 (0.001, 2 seeds) |
| merge: task_arithmetic | 0.973 (0.002, 3 seeds) |
| merge: model_stock | 0.472 (0.002, 3 seeds) |
| merge: slerp | 0.974 (0.003, 3 seeds) |
| merge: arcee_fusion | 0.963 (0.003, 3 seeds) |
| MoE (mergekit-moe) | 0.975 (0.001, 3 seeds) |
| MoE, prompt_mean router | 0.975 (0.002, 3 seeds) |
| MoE, random router (control) | 0.949 (0.000, 3 seeds) |

### Mistral 7B: XSum

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |
|---|---|---|---|---|
| Base model (untrained) | 0.238 | 0.053 | 0.175 | 22.3 |
| Expert (trained on that task) | 0.375 (0.004) | 0.142 (0.003) | 0.290 (0.003) | 24.5 (0.1) |
| merge: linear | 0.325 (0.006, 2 seeds) | 0.112 (0.003, 2 seeds) | 0.252 (0.006, 2 seeds) | 20.3 (0.2, 2 seeds) |
| merge: ties | 0.059 (0.027, 2 seeds) | 0.010 (0.004, 2 seeds) | 0.050 (0.020, 2 seeds) | 5.1 (2.8, 2 seeds) |
| merge: dare_ties | 0.313 (0.006, 2 seeds) | 0.104 (0.003, 2 seeds) | 0.244 (0.006, 2 seeds) | 19.1 (0.3, 2 seeds) |
| merge: task_arithmetic | 0.329 (0.006, 3 seeds) | 0.114 (0.003, 3 seeds) | 0.255 (0.005, 3 seeds) | 20.6 (0.3, 3 seeds) |
| merge: model_stock | 0.237 (0.001, 3 seeds) | 0.053 (0.000, 3 seeds) | 0.175 (0.001, 3 seeds) | 22.4 (0.1, 3 seeds) |
| merge: slerp | 0.329 (0.005, 3 seeds) | 0.115 (0.002, 3 seeds) | 0.255 (0.004, 3 seeds) | 20.4 (0.3, 3 seeds) |
| merge: arcee_fusion | 0.172 (0.041, 3 seeds) | 0.049 (0.012, 3 seeds) | 0.137 (0.033, 3 seeds) | 10.6 (1.1, 3 seeds) |
| MoE (mergekit-moe) | 0.349 (0.016, 3 seeds) | 0.134 (0.007, 3 seeds) | 0.273 (0.014, 3 seeds) | 23.6 (3.4, 3 seeds) |
| MoE, prompt_mean router | 0.287 (0.027, 3 seeds) | 0.101 (0.012, 3 seeds) | 0.222 (0.019, 3 seeds) | 17.7 (1.9, 3 seeds) |
| MoE, random router (control) | 0.327 (0.031, 3 seeds) | 0.117 (0.015, 3 seeds) | 0.253 (0.025, 3 seeds) | 21.7 (0.9, 3 seeds) |

### Llama 3.1 8B: Crime

| median (IQR) | Accuracy | F1 |
|---|---|---|
| Base model (untrained) | 0.743 | 0.592 |
| Expert (trained on that task) | 0.985 (0.004) | 0.983 (0.004) |
| merge: linear | 0.898 (0.000, 2 seeds) | 0.874 (0.000, 2 seeds) |
| merge: ties | 0.945 (0.002, 2 seeds) | 0.935 (0.003, 2 seeds) |
| merge: dare_ties | 0.900 (0.002, 2 seeds) | 0.877 (0.002, 2 seeds) |
| merge: task_arithmetic | 0.898 (0.000, 2 seeds) | 0.874 (0.000, 2 seeds) |
| merge: model_stock | 0.751 (0.001, 2 seeds) | 0.609 (0.003, 2 seeds) |
| merge: slerp | 0.898 (0.003, 3 seeds) | 0.874 (0.003, 3 seeds) |
| merge: arcee_fusion | 0.967 (0.001, 2 seeds) | 0.963 (0.002, 2 seeds) |
| MoE (mergekit-moe) | 0.923 (0.007, 3 seeds) | 0.907 (0.009, 3 seeds) |
| MoE, prompt_mean router | 0.974 (0.009, 3 seeds) | 0.970 (0.011, 3 seeds) |
| MoE, random router (control) | 0.902 (0.006, 3 seeds) | 0.879 (0.007, 3 seeds) |

### Llama 3.1 8B: Wine

| median (IQR) | MAE | RMSE | Pearson |
|---|---|---|---|
| Base model (untrained) | 4.356 | 4.912 | 0.646 |
| Expert (trained on that task) | 1.140 (0.016) | 1.523 (0.004) | 0.877 (0.001) |
| merge: linear | 1.251 (0.014, 3 seeds) | 1.658 (0.007, 3 seeds) | 0.851 (0.002, 3 seeds) |
| merge: ties | 1.187 (0.015, 2 seeds) | 1.568 (0.015, 2 seeds) | 0.867 (0.002, 2 seeds) |
| merge: dare_ties | 1.243 (0.001, 2 seeds) | 1.646 (0.003, 2 seeds) | 0.852 (0.000, 2 seeds) |
| merge: task_arithmetic | 1.252 (0.012, 3 seeds) | 1.652 (0.009, 3 seeds) | 0.851 (0.001, 3 seeds) |
| merge: model_stock | 4.166 (0.003, 2 seeds) | 4.706 (0.003, 2 seeds) | 0.674 (0.001, 2 seeds) |
| merge: slerp | 1.244 (0.003, 2 seeds) | 1.650 (0.000, 2 seeds) | 0.851 (0.001, 2 seeds) |
| merge: arcee_fusion | 1.348 (0.014, 2 seeds) | 1.779 (0.020, 2 seeds) | 0.833 (0.001, 2 seeds) |
| MoE (mergekit-moe) | 1.191 (0.021, 3 seeds) | 1.580 (0.026, 3 seeds) | 0.872 (0.002, 3 seeds) |
| MoE, prompt_mean router | 1.144 (0.008, 3 seeds) | 1.507 (0.007, 3 seeds) | 0.879 (0.002, 3 seeds) |
| MoE, random router (control) | 1.282 (0.062, 3 seeds) | 1.688 (0.076, 3 seeds) | 0.846 (0.010, 3 seeds) |

### Llama 3.1 8B: CLINC150

| median (IQR) | Accuracy |
|---|---|
| Base model (untrained) | 0.640 |
| Expert (trained on that task) | 0.979 (0.001) |
| merge: linear | 0.965 (0.002, 2 seeds) |
| merge: ties | 0.979 (0.001, 3 seeds) |
| merge: dare_ties | 0.967 (0.003, 2 seeds) |
| merge: task_arithmetic | 0.965 (0.002, 2 seeds) |
| merge: model_stock | 0.651 (0.001, 2 seeds) |
| merge: slerp | 0.965 (0.002, 2 seeds) |
| merge: arcee_fusion | 0.949 (0.003, 3 seeds) |
| MoE (mergekit-moe) | 0.976 (0.001, 3 seeds) |
| MoE, prompt_mean router | 0.978 (0.001, 3 seeds) |
| MoE, random router (control) | 0.958 (0.012, 3 seeds) |

### Llama 3.1 8B: XSum

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |
|---|---|---|---|---|
| Base model (untrained) | 0.289 | 0.084 | 0.218 | 21.9 |
| Expert (trained on that task) | 0.378 (0.001) | 0.149 (0.002) | 0.299 (0.001) | 20.6 (0.2) |
| merge: linear | 0.338 (0.002, 2 seeds) | 0.116 (0.001, 2 seeds) | 0.261 (0.001, 2 seeds) | 20.6 (0.2, 2 seeds) |
| merge: ties | 0.260 (0.005, 2 seeds) | 0.073 (0.001, 2 seeds) | 0.210 (0.002, 2 seeds) | 12.9 (0.4, 2 seeds) |
| merge: dare_ties | 0.335 (0.001, 3 seeds) | 0.114 (0.001, 3 seeds) | 0.258 (0.001, 3 seeds) | 20.2 (0.5, 3 seeds) |
| merge: task_arithmetic | 0.338 (0.002, 3 seeds) | 0.118 (0.001, 3 seeds) | 0.262 (0.001, 3 seeds) | 20.4 (0.3, 3 seeds) |
| merge: model_stock | 0.289 (0.001, 2 seeds) | 0.085 (0.000, 2 seeds) | 0.219 (0.000, 2 seeds) | 21.7 (0.0, 2 seeds) |
| merge: slerp | 0.338 (0.002, 3 seeds) | 0.116 (0.002, 3 seeds) | 0.261 (0.002, 3 seeds) | 20.4 (0.4, 3 seeds) |
| merge: arcee_fusion | 0.312 (0.003, 3 seeds) | 0.098 (0.001, 3 seeds) | 0.237 (0.002, 3 seeds) | 19.6 (0.6, 3 seeds) |
| MoE (mergekit-moe) | 0.356 (0.001, 3 seeds) | 0.134 (0.002, 3 seeds) | 0.279 (0.002, 3 seeds) | 21.5 (0.2, 3 seeds) |
| MoE, prompt_mean router | 0.335 (0.005, 3 seeds) | 0.119 (0.004, 3 seeds) | 0.260 (0.004, 3 seeds) | 19.8 (0.3, 3 seeds) |
| MoE, random router (control) | 0.335 (0.006, 3 seeds) | 0.115 (0.003, 3 seeds) | 0.257 (0.005, 3 seeds) | 20.8 (0.9, 3 seeds) |
