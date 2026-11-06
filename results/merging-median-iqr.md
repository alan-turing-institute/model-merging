# Merging results by median and interquartile range

Generated 2026-11-06 by `evaluate/make_merging_report.py` from the per-model JSONs in `results/` (as of commit 75e72e9).

Contents: data-parallel merging of crime and XSum (Gemma 3 4B); task merging (Gemma 3 4B, Mistral 7B, Llama 3.1 8B); whether the summary statistic changes the verdict; outlying seeds; an appendix with every task-merging metric.

Each cell is the median over seeds 1-5 with the interquartile range in brackets, *median (IQR)*, as each table's corner says; the base model is evaluated once. Quartiles interpolate linearly between the sorted values (Python's `statistics.quantiles` with `method="inclusive"`, numpy's default), so with five seeds Q1 and Q3 are the second and fourth smallest values and the IQR, Q3 - Q1, is the full width of the middle three seeds, not a ± half-width. For five seeds that scatter normally it averages 0.99 standard deviations, about 1.05 times the sample SD the mean ± SD reports give. Q1 and Q3 are single seeds, so the IQR is itself noisy, and (0.000) means only that the second and fourth seeds agree at the precision shown.

One seed that fails can move the median and IQR only as far as a neighbouring seed's value, whereas it can move the mean and standard deviation without limit. The price is that the median and IQR ignore the best and worst seed, so a method that fails on one seed in five can look reliable: cells with such a seed are marked † and listed under *Outlying seeds*.

The data-parallel tables are also in `results/crime-data-parallel-gemma.md` and `results/xsum-data-parallel-gemma.md` as mean ± SD. Those are built from the `.eval` logs, and these JSONs reproduce every row of them (`--stat mean`), so any difference comes from the statistic alone. Task merging has no mean ± SD report; `make_merging_report.py --stat mean` writes this one that way.

Every arm is repeated with seeds 1-5. The seed sets the training data order (for the LoRA experts also the adapter initialisation and dropout) and mergekit's random seed, which dare_ties uses to choose the weights it drops and arcee_fusion to sample the tensors its thresholds are estimated from.

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

Four experts, one per task, each trained on all of its task's data (MLP-only: LoRA for crime, wine and CLINC150; self-distillation for XSum), are merged into one model, which is evaluated on all four tasks. Each method merges the same four experts. The MoE (Mistral and Llama only: mergekit has no MoE path for Gemma) routes each token, at every layer, to one expert's MLP; its attention, embeddings and norms are the base model's. Test sets: Crime, 1,077 test posts; Wine, 1,000 test reviews (error in points: lower is better); CLINC150, 4,500 test queries, labels prompt; XSum, 1,000 test articles, long prompt.

*Recovery R* is the share of the expert's gain over the base model that a merge keeps, for one task and seed: (merge - base) / (expert - base), with the expert of the same seed, on the task's main metric (crime accuracy, wine MAE, CLINC150 accuracy, XSum ROUGE-1: the columns of each model's first table). 1 is as good as the expert, 0 no better than the untrained model, and below 0 worse than it. *All four* is the mean of a seed's four R values, then the median over seeds; it is a mean because it is a score, and a median of four would ignore the best and the worst task.

### Gemma 3 4B (`google/gemma-3-4b-it`)

| median (IQR) | Crime Accuracy | Wine MAE | CLINC150 Accuracy | XSum ROUGE-1 |
|---|---|---|---|---|
| Base model (untrained) | 0.755 | 2.376 | 0.704 | 0.247 |
| Expert (trained on that task) | 0.980 (0.002) | 1.179 (0.005) | 0.978 (0.000) | 0.362 (0.001) |
| merge: linear | 0.892 (0.006) | 1.407 (0.048) | 0.961 (0.005) | 0.301 (0.006) |
| merge: ties | 0.940 (0.007) | 1.222 (0.029) † | 0.977 (0.001) | 0.260 (0.018) |
| merge: dare_ties | 0.894 (0.008) | 1.399 (0.034) | 0.963 (0.003) | 0.299 (0.003) |
| merge: task_arithmetic | 0.892 (0.007) | 1.401 (0.040) | 0.961 (0.005) | 0.301 (0.005) |
| merge: model_stock | 0.760 (0.002) | 2.377 (0.029) | 0.717 (0.000) | 0.247 (0.000) |
| merge: slerp | 0.892 (0.006) | 1.410 (0.027) | 0.961 (0.005) | 0.302 (0.007) |
| merge: arcee_fusion | 0.924 (0.014) | 1.650 (0.084) | 0.936 (0.006) | 0.273 (0.006) |

Recovery R (median over seeds):

| median (IQR) | Crime | Wine | CLINC150 | XSum | All four |
|---|---|---|---|---|---|
| merge: linear | 0.61 (0.02) | 0.80 (0.04) | 0.94 (0.02) | 0.49 (0.06) | 0.70 (0.02) |
| merge: ties | 0.82 (0.03) | 0.95 (0.02) † | 1.00 (0.01) | 0.12 (0.15) | 0.72 (0.03) † |
| merge: dare_ties | 0.62 (0.03) | 0.80 (0.03) | 0.95 (0.01) | 0.47 (0.02) | 0.71 (0.01) |
| merge: task_arithmetic | 0.61 (0.03) | 0.80 (0.03) | 0.94 (0.02) | 0.48 (0.04) | 0.71 (0.02) |
| merge: model_stock | 0.02 (0.01) | 0.00 (0.02) | 0.05 (0.00) | 0.00 (0.00) | 0.02 (0.00) |
| merge: slerp | 0.61 (0.02) | 0.79 (0.02) | 0.94 (0.02) | 0.48 (0.05) | 0.70 (0.02) |
| merge: arcee_fusion | 0.75 (0.07) | 0.60 (0.08) | 0.85 (0.02) | 0.23 (0.05) | 0.60 (0.01) |

### Mistral 7B (`mistralai/Mistral-7B-Instruct-v0.3`)

| median (IQR) | Crime Accuracy | Wine MAE | CLINC150 Accuracy | XSum ROUGE-1 |
|---|---|---|---|---|
| Base model (untrained) | 0.774 | 1.676 | 0.454 | 0.238 |
| Expert (trained on that task) | 0.983 (0.001) | 1.137 (0.016) | 0.975 (0.002) | 0.375 (0.001) |
| merge: linear | 0.939 (0.005) | 1.267 (0.004) | 0.973 (0.002) | 0.328 (0.008) |
| merge: ties | 0.955 (0.016) | 1.181 (0.015) | 0.978 (0.002) | 0.052 (0.024) |
| merge: dare_ties | 0.936 (0.006) | 1.269 (0.002) | 0.973 (0.002) | 0.306 (0.012) |
| merge: task_arithmetic | 0.940 (0.008) | 1.261 (0.015) | 0.973 (0.002) | 0.328 (0.009) |
| merge: model_stock | 0.780 (0.003) | 1.661 (0.004) | 0.472 (0.002) | 0.237 (0.002) |
| merge: slerp | 0.939 (0.006) | 1.271 (0.011) | 0.973 (0.002) | 0.327 (0.007) |
| merge: arcee_fusion | 0.979 (0.004) | 1.337 (0.018) | 0.962 (0.000) | 0.171 (0.043) |
| MoE (mergekit-moe) | 0.906 (0.045) | 1.194 (0.022) | 0.975 (0.003) | 0.351 (0.019) |

Recovery R (median over seeds):

| median (IQR) | Crime | Wine | CLINC150 | XSum | All four |
|---|---|---|---|---|---|
| merge: linear | 0.78 (0.02) † | 0.75 (0.03) | 1.00 (0.00) | 0.63 (0.04) | 0.79 (0.02) |
| merge: ties | 0.87 (0.07) | 0.92 (0.04) | 1.00 (0.00) | -1.35 (0.20) | 0.35 (0.04) |
| merge: dare_ties | 0.77 (0.02) † | 0.75 (0.03) | 0.99 (0.00) | 0.50 (0.07) | 0.77 (0.03) |
| merge: task_arithmetic | 0.79 (0.03) | 0.76 (0.05) | 1.00 (0.00) | 0.63 (0.05) | 0.80 (0.03) |
| merge: model_stock | 0.03 (0.01) | 0.03 (0.01) | 0.03 (0.00) | 0.00 (0.01) | 0.02 (0.01) |
| merge: slerp | 0.78 (0.02) | 0.75 (0.03) | 1.00 (0.00) | 0.64 (0.03) | 0.79 (0.03) |
| merge: arcee_fusion | 0.97 (0.02) | 0.63 (0.04) | 0.98 (0.00) | -0.48 (0.32) | 0.52 (0.07) |
| MoE (mergekit-moe) | 0.63 (0.21) | 0.88 (0.02) | 1.00 (0.01) | 0.82 (0.12) † | 0.81 (0.06) |

### Llama 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`)

| median (IQR) | Crime Accuracy | Wine MAE | CLINC150 Accuracy | XSum ROUGE-1 |
|---|---|---|---|---|
| Base model (untrained) | 0.743 | 4.356 | 0.640 | 0.289 |
| Expert (trained on that task) | 0.985 (0.004) | 1.140 (0.016) | 0.979 (0.001) | 0.380 (0.001) |
| merge: linear | 0.901 (0.002) | 1.243 (0.019) | 0.963 (0.005) | 0.336 (0.004) |
| merge: ties | 0.946 (0.002) | 1.180 (0.009) | 0.979 (0.001) | 0.268 (0.004) |
| merge: dare_ties | 0.901 (0.000) | 1.242 (0.023) | 0.964 (0.006) | 0.335 (0.003) |
| merge: task_arithmetic | 0.901 (0.002) | 1.245 (0.021) | 0.963 (0.005) | 0.336 (0.004) |
| merge: model_stock | 0.750 (0.004) | 4.165 (0.089) | 0.652 (0.002) | 0.289 (0.000) |
| merge: slerp | 0.900 (0.005) | 1.249 (0.011) | 0.963 (0.006) | 0.336 (0.001) |
| merge: arcee_fusion | 0.962 (0.011) | 1.363 (0.023) | 0.949 (0.004) | 0.310 (0.004) |
| MoE (mergekit-moe) | 0.929 (0.013) | 1.184 (0.034) | 0.976 (0.001) | 0.356 (0.002) |

Recovery R (median over seeds):

| median (IQR) | Crime | Wine | CLINC150 | XSum | All four |
|---|---|---|---|---|---|
| merge: linear | 0.66 (0.02) | 0.97 (0.01) | 0.95 (0.01) | 0.52 (0.04) | 0.78 (0.01) |
| merge: ties | 0.84 (0.01) | 0.99 (0.00) | 1.00 (0.01) | -0.23 (0.05) | 0.65 (0.01) |
| merge: dare_ties | 0.65 (0.01) | 0.97 (0.01) | 0.95 (0.02) | 0.51 (0.03) | 0.77 (0.01) |
| merge: task_arithmetic | 0.65 (0.00) | 0.97 (0.01) | 0.95 (0.01) | 0.52 (0.04) | 0.78 (0.01) |
| merge: model_stock | 0.03 (0.01) | 0.06 (0.03) | 0.04 (0.01) | 0.01 (0.00) | 0.03 (0.01) |
| merge: slerp | 0.65 (0.02) | 0.97 (0.01) | 0.95 (0.01) | 0.53 (0.03) | 0.78 (0.01) |
| merge: arcee_fusion | 0.90 (0.03) | 0.93 (0.01) | 0.91 (0.02) | 0.23 (0.05) | 0.74 (0.01) |
| MoE (mergekit-moe) | 0.77 (0.05) | 0.99 (0.00) | 0.99 (0.00) | 0.74 (0.04) | 0.88 (0.02) |

## Does the summary statistic change the verdict?

In 8 of the 9 comparisons the best merge by the median is also the best by the mean, ties at the precision shown counting as best; it changes in Task merging, Gemma 3 4B. model_stock is the worst merge, or tied worst, in every comparison under both statistics.

The main metric of each comparison is accuracy (crime), ROUGE-1 (XSum) and R over all four tasks (task merging). Methods are tied when they agree at the precision shown, and tied methods are listed together, the better unrounded first. *Next* is the methods with the second-best median. *Wins* counts the seeds on which the first best method beats each of them, out of the seeds both have: every method of a seed merges the same experts, so the comparison is paired. With five seeds neither the medians nor the win counts are a significance test; they show how consistent a lead is. *Same best* says whether the best by the median and by the mean have a method in common.

| Comparison | Best by median | Next by median | Wins | Best by mean | Same best |
|---|---|---|---|---|---|
| Data-parallel Crime, N = 2 | ties 0.968 | dare_ties 0.965 | 4 of 5 | ties 0.968 | yes |
| Data-parallel Crime, N = 4 | ties 0.956 | dare_ties = task_arithmetic = slerp 0.946 | 5 of 5, 5 of 5, 5 of 5 | ties 0.957 | yes |
| Data-parallel Crime, N = 8 | ties 0.935 | arcee_fusion 0.920 | 5 of 5 | ties 0.935 | yes |
| Data-parallel XSum, N = 2 | ties 0.360 | task_arithmetic = linear = arcee_fusion 0.359 | 3 of 5, 3 of 5, 4 of 5 | ties 0.360 | yes |
| Data-parallel XSum, N = 4 | linear 0.357 | task_arithmetic 0.356 | 4 of 5 | linear = task_arithmetic = ties = dare_ties = slerp 0.355 | yes |
| Data-parallel XSum, N = 8 | arcee_fusion 0.349 | ties = dare_ties = linear = slerp = task_arithmetic 0.348 | 3 of 5, 4 of 5, 5 of 5, 4 of 5, 5 of 5 | arcee_fusion 0.349 | yes |
| Task merging, Gemma 3 4B | ties 0.72 | dare_ties = task_arithmetic 0.71 | 3 of 5, 3 of 5 | task_arithmetic = slerp = dare_ties = linear 0.71 | no |
| Task merging, Mistral 7B | MoE (mergekit-moe) 0.81 | task_arithmetic 0.80 | 4 of 5 | MoE (mergekit-moe) = task_arithmetic = slerp = linear 0.79 | yes |
| Task merging, Llama 3.1 8B | MoE (mergekit-moe) 0.88 | task_arithmetic = linear = slerp 0.78 | 5 of 5, 5 of 5, 5 of 5 | MoE (mergekit-moe) 0.88 | yes |

## Outlying seeds

Cells in which a seed lies more than 3 IQRs beyond the nearer quartile, taking as the IQR the larger of the cell's own and the median IQR of that metric over every arm of the study (with five seeds a cell's own IQR is often a unit or two of the last digit, which would make ordinary noise outlying): 12 of the 525 distinct five-seed cells, the appendix included. They are marked † in the tables. The median and IQR barely move for such a seed, which is the point of them, but they also hide it, and a method that fails on one seed in five is less reliable than its median suggests. The outlying values are in bold; *Mean ± SD* is computed as in the mean ± SD reports.

| Where | Row | Metric | Seeds 1-5 | Mean ± SD | Median (IQR) |
|---|---|---|---|---|---|
| Data-parallel XSum, N = 2 | Experts (each on 1/2 of the data) | ROUGE-1 | **0.354**, 0.359, 0.359, 0.360, 0.358 | 0.358 ± 0.002 | 0.359 (0.001) |
| Data-parallel XSum, N = 4 | merge: slerp | ROUGE-1 | 0.356, **0.351**, 0.356, 0.355, 0.355 | 0.355 ± 0.002 | 0.355 (0.000) |
| Task merging, Gemma 3 4B | merge: ties | R Wine | 0.95, **-0.67**, 0.94, 0.99, 0.97 | 0.64 ± 0.73 | 0.95 (0.02) |
| Task merging, Gemma 3 4B | merge: ties | Wine MAE | 1.222, **3.173**, 1.247, 1.211, 1.218 | 1.614 ± 0.872 | 1.222 (0.029) |
| Task merging, Gemma 3 4B | merge: ties | Wine Pearson | 0.860, **0.052**, 0.857, 0.860, 0.861 | 0.698 ± 0.361 | 0.860 (0.004) |
| Task merging, Gemma 3 4B | merge: ties | Wine RMSE | 1.614, **60.991**, 1.632, 1.611, 1.601 | 13.490 ± 26.554 | 1.614 (0.021) |
| Task merging, Gemma 3 4B | merge: dare_ties | XSum ROUGE-L | 0.229, 0.230, 0.228, **0.238**, 0.228 | 0.231 ± 0.004 | 0.229 (0.001) |
| Task merging, Gemma 3 4B | merge: arcee_fusion | XSum Words | 19.5, **17.2**, 19.2, 19.5, 19.5 | 19.0 ± 1.0 | 19.5 (0.3) |
| Task merging, Gemma 3 4B | merge: ties | R all four | 0.72, **0.30**, 0.74, 0.76, 0.71 | 0.64 ± 0.19 | 0.72 (0.03) |
| Task merging, Mistral 7B | merge: linear | R Crime | 0.77, **0.71**, 0.78, 0.79, 0.83 | 0.77 ± 0.05 | 0.78 (0.02) |
| Task merging, Mistral 7B | merge: dare_ties | R Crime | 0.75, **0.70**, 0.77, 0.77, **0.85** | 0.77 ± 0.05 | 0.77 (0.02) |
| Task merging, Mistral 7B | MoE (mergekit-moe) | R XSum | 0.88, 0.70, 0.82, 0.82, **0.32** | 0.71 ± 0.23 | 0.82 (0.12) |

By arm and seed, with the arm's score on that seed and on its other four, on the main metric of its comparison (as in the verdict above), so that a seed can be judged as a whole: a seed outlying on one metric may still be weak, or fine, overall.

| Where | Row | Seed | Outlying on | Score on that seed | Other seeds |
|---|---|---|---|---|---|
| Data-parallel XSum, N = 2 | Experts (each on 1/2 of the data) | 1 | ROUGE-1 | ROUGE-1 0.354 | 0.358 to 0.360 |
| Data-parallel XSum, N = 4 | merge: slerp | 2 | ROUGE-1 | ROUGE-1 0.351 | 0.355 to 0.356 |
| Task merging, Gemma 3 4B | merge: ties | 2 | R Wine, Wine MAE, Wine Pearson, Wine RMSE, R all four | R all four 0.30 | 0.71 to 0.76 |
| Task merging, Gemma 3 4B | merge: dare_ties | 4 | XSum ROUGE-L | R all four 0.73 | 0.70 to 0.72 |
| Task merging, Gemma 3 4B | merge: arcee_fusion | 2 | XSum Words | R all four 0.58 | 0.60 to 0.61 |
| Task merging, Mistral 7B | merge: linear | 2 | R Crime | R all four 0.76 | 0.77 to 0.81 |
| Task merging, Mistral 7B | merge: dare_ties | 2 | R Crime | R all four 0.74 | 0.75 to 0.78 |
| Task merging, Mistral 7B | merge: dare_ties | 5 | R Crime | R all four 0.78 | 0.74 to 0.78 |
| Task merging, Mistral 7B | MoE (mergekit-moe) | 5 | R XSum | R all four 0.63 | 0.79 to 0.87 |

## Appendix: every metric, task merging

### Gemma 3 4B: Crime

| median (IQR) | Accuracy | F1 |
|---|---|---|
| Base model (untrained) | 0.755 | 0.624 |
| Expert (trained on that task) | 0.980 (0.002) | 0.977 (0.002) |
| merge: linear | 0.892 (0.006) | 0.865 (0.009) |
| merge: ties | 0.940 (0.007) | 0.927 (0.006) |
| merge: dare_ties | 0.894 (0.008) | 0.867 (0.013) |
| merge: task_arithmetic | 0.892 (0.007) | 0.865 (0.011) |
| merge: model_stock | 0.760 (0.002) | 0.636 (0.004) |
| merge: slerp | 0.892 (0.006) | 0.865 (0.010) |
| merge: arcee_fusion | 0.924 (0.014) | 0.908 (0.016) |

### Gemma 3 4B: Wine

| median (IQR) | MAE | RMSE | Pearson |
|---|---|---|---|
| Base model (untrained) | 2.376 | 3.322 | 0.671 |
| Expert (trained on that task) | 1.179 (0.005) | 1.573 (0.030) | 0.870 (0.007) |
| merge: linear | 1.407 (0.048) | 1.852 (0.155) | 0.831 (0.029) |
| merge: ties | 1.222 (0.029) † | 1.614 (0.021) † | 0.860 (0.004) † |
| merge: dare_ties | 1.399 (0.034) | 1.853 (0.115) | 0.832 (0.022) |
| merge: task_arithmetic | 1.401 (0.040) | 1.845 (0.142) | 0.833 (0.025) |
| merge: model_stock | 2.377 (0.029) | 3.316 (0.041) | 0.672 (0.003) |
| merge: slerp | 1.410 (0.027) | 1.877 (0.137) | 0.822 (0.028) |
| merge: arcee_fusion | 1.650 (0.084) | 2.413 (0.121) | 0.745 (0.011) |

### Gemma 3 4B: CLINC150

| median (IQR) | Accuracy |
|---|---|
| Base model (untrained) | 0.704 |
| Expert (trained on that task) | 0.978 (0.000) |
| merge: linear | 0.961 (0.005) |
| merge: ties | 0.977 (0.001) |
| merge: dare_ties | 0.963 (0.003) |
| merge: task_arithmetic | 0.961 (0.005) |
| merge: model_stock | 0.717 (0.000) |
| merge: slerp | 0.961 (0.005) |
| merge: arcee_fusion | 0.936 (0.006) |

### Gemma 3 4B: XSum

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |
|---|---|---|---|---|
| Base model (untrained) | 0.247 | 0.055 | 0.181 | 19.6 |
| Expert (trained on that task) | 0.362 (0.001) | 0.136 (0.002) | 0.285 (0.003) | 19.2 (0.3) |
| merge: linear | 0.301 (0.006) | 0.090 (0.004) | 0.232 (0.003) | 19.7 (0.7) |
| merge: ties | 0.260 (0.018) | 0.068 (0.008) | 0.204 (0.012) | 15.2 (2.1) |
| merge: dare_ties | 0.299 (0.003) | 0.089 (0.002) | 0.229 (0.001) † | 19.9 (0.7) |
| merge: task_arithmetic | 0.301 (0.005) | 0.090 (0.003) | 0.231 (0.003) | 19.7 (0.6) |
| merge: model_stock | 0.247 (0.000) | 0.055 (0.001) | 0.183 (0.001) | 19.7 (0.1) |
| merge: slerp | 0.302 (0.007) | 0.090 (0.004) | 0.232 (0.003) | 19.7 (0.7) |
| merge: arcee_fusion | 0.273 (0.006) | 0.070 (0.003) | 0.206 (0.005) | 19.5 (0.3) † |

### Mistral 7B: Crime

| median (IQR) | Accuracy | F1 |
|---|---|---|
| Base model (untrained) | 0.774 | 0.682 |
| Expert (trained on that task) | 0.983 (0.001) | 0.981 (0.001) |
| merge: linear | 0.939 (0.005) | 0.926 (0.007) |
| merge: ties | 0.955 (0.016) | 0.949 (0.020) |
| merge: dare_ties | 0.936 (0.006) | 0.922 (0.007) |
| merge: task_arithmetic | 0.940 (0.008) | 0.927 (0.011) |
| merge: model_stock | 0.780 (0.003) | 0.695 (0.005) |
| merge: slerp | 0.939 (0.006) | 0.926 (0.009) |
| merge: arcee_fusion | 0.979 (0.004) | 0.976 (0.004) |
| MoE (mergekit-moe) | 0.906 (0.045) | 0.890 (0.046) |

### Mistral 7B: Wine

| median (IQR) | MAE | RMSE | Pearson |
|---|---|---|---|
| Base model (untrained) | 1.676 | 2.255 | 0.740 |
| Expert (trained on that task) | 1.137 (0.016) | 1.511 (0.006) | 0.878 (0.002) |
| merge: linear | 1.267 (0.004) | 1.681 (0.005) | 0.864 (0.002) |
| merge: ties | 1.181 (0.015) | 1.556 (0.015) | 0.871 (0.002) |
| merge: dare_ties | 1.269 (0.002) | 1.675 (0.011) | 0.865 (0.003) |
| merge: task_arithmetic | 1.261 (0.015) | 1.675 (0.020) | 0.865 (0.001) |
| merge: model_stock | 1.661 (0.004) | 2.219 (0.022) | 0.746 (0.004) |
| merge: slerp | 1.271 (0.011) | 1.684 (0.012) | 0.863 (0.002) |
| merge: arcee_fusion | 1.337 (0.018) | 1.762 (0.022) | 0.846 (0.003) |
| MoE (mergekit-moe) | 1.194 (0.022) | 1.592 (0.016) | 0.868 (0.002) |

### Mistral 7B: CLINC150

| median (IQR) | Accuracy |
|---|---|
| Base model (untrained) | 0.454 |
| Expert (trained on that task) | 0.975 (0.002) |
| merge: linear | 0.973 (0.002) |
| merge: ties | 0.978 (0.002) |
| merge: dare_ties | 0.973 (0.002) |
| merge: task_arithmetic | 0.973 (0.002) |
| merge: model_stock | 0.472 (0.002) |
| merge: slerp | 0.973 (0.002) |
| merge: arcee_fusion | 0.962 (0.000) |
| MoE (mergekit-moe) | 0.975 (0.003) |

### Mistral 7B: XSum

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |
|---|---|---|---|---|
| Base model (untrained) | 0.238 | 0.053 | 0.175 | 22.3 |
| Expert (trained on that task) | 0.375 (0.001) | 0.143 (0.001) | 0.292 (0.003) | 24.5 (0.2) |
| merge: linear | 0.328 (0.008) | 0.113 (0.005) | 0.253 (0.005) | 20.4 (0.6) |
| merge: ties | 0.052 (0.024) | 0.009 (0.004) | 0.048 (0.021) | 2.8 (1.6) |
| merge: dare_ties | 0.306 (0.012) | 0.100 (0.007) | 0.241 (0.007) | 18.8 (0.7) |
| merge: task_arithmetic | 0.328 (0.009) | 0.113 (0.005) | 0.254 (0.005) | 20.4 (0.5) |
| merge: model_stock | 0.237 (0.002) | 0.053 (0.000) | 0.175 (0.000) | 22.3 (0.1) |
| merge: slerp | 0.327 (0.007) | 0.113 (0.004) | 0.253 (0.004) | 20.2 (0.7) |
| merge: arcee_fusion | 0.171 (0.043) | 0.047 (0.012) | 0.135 (0.034) | 10.6 (1.9) |
| MoE (mergekit-moe) | 0.351 (0.019) | 0.134 (0.011) | 0.273 (0.016) | 23.9 (4.5) |

### Llama 3.1 8B: Crime

| median (IQR) | Accuracy | F1 |
|---|---|---|
| Base model (untrained) | 0.743 | 0.592 |
| Expert (trained on that task) | 0.985 (0.004) | 0.983 (0.004) |
| merge: linear | 0.901 (0.002) | 0.877 (0.003) |
| merge: ties | 0.946 (0.002) | 0.936 (0.003) |
| merge: dare_ties | 0.901 (0.000) | 0.878 (0.002) |
| merge: task_arithmetic | 0.901 (0.002) | 0.876 (0.002) |
| merge: model_stock | 0.750 (0.004) | 0.608 (0.008) |
| merge: slerp | 0.900 (0.005) | 0.875 (0.005) |
| merge: arcee_fusion | 0.962 (0.011) | 0.956 (0.015) |
| MoE (mergekit-moe) | 0.929 (0.013) | 0.916 (0.018) |

### Llama 3.1 8B: Wine

| median (IQR) | MAE | RMSE | Pearson |
|---|---|---|---|
| Base model (untrained) | 4.356 | 4.912 | 0.646 |
| Expert (trained on that task) | 1.140 (0.016) | 1.523 (0.004) | 0.877 (0.001) |
| merge: linear | 1.243 (0.019) | 1.647 (0.019) | 0.852 (0.002) |
| merge: ties | 1.180 (0.009) | 1.563 (0.009) | 0.868 (0.002) |
| merge: dare_ties | 1.242 (0.023) | 1.645 (0.025) | 0.852 (0.004) |
| merge: task_arithmetic | 1.245 (0.021) | 1.647 (0.013) | 0.853 (0.003) |
| merge: model_stock | 4.165 (0.089) | 4.704 (0.099) | 0.671 (0.011) |
| merge: slerp | 1.249 (0.011) | 1.652 (0.014) | 0.851 (0.001) |
| merge: arcee_fusion | 1.363 (0.023) | 1.775 (0.033) | 0.833 (0.007) |
| MoE (mergekit-moe) | 1.184 (0.034) | 1.564 (0.034) | 0.870 (0.003) |

### Llama 3.1 8B: CLINC150

| median (IQR) | Accuracy |
|---|---|
| Base model (untrained) | 0.640 |
| Expert (trained on that task) | 0.979 (0.001) |
| merge: linear | 0.963 (0.005) |
| merge: ties | 0.979 (0.001) |
| merge: dare_ties | 0.964 (0.006) |
| merge: task_arithmetic | 0.963 (0.005) |
| merge: model_stock | 0.652 (0.002) |
| merge: slerp | 0.963 (0.006) |
| merge: arcee_fusion | 0.949 (0.004) |
| MoE (mergekit-moe) | 0.976 (0.001) |

### Llama 3.1 8B: XSum

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |
|---|---|---|---|---|
| Base model (untrained) | 0.289 | 0.084 | 0.218 | 21.9 |
| Expert (trained on that task) | 0.380 (0.001) | 0.151 (0.002) | 0.300 (0.002) | 20.3 (0.2) |
| merge: linear | 0.336 (0.004) | 0.115 (0.002) | 0.259 (0.002) | 20.8 (0.4) |
| merge: ties | 0.268 (0.004) | 0.077 (0.000) | 0.216 (0.002) | 13.1 (0.9) |
| merge: dare_ties | 0.335 (0.003) | 0.113 (0.001) | 0.257 (0.002) | 20.8 (0.7) |
| merge: task_arithmetic | 0.336 (0.004) | 0.116 (0.002) | 0.260 (0.002) | 20.8 (0.4) |
| merge: model_stock | 0.289 (0.000) | 0.084 (0.000) | 0.219 (0.000) | 21.8 (0.0) |
| merge: slerp | 0.336 (0.001) | 0.116 (0.001) | 0.260 (0.001) | 20.8 (0.5) |
| merge: arcee_fusion | 0.310 (0.004) | 0.099 (0.002) | 0.238 (0.002) | 20.2 (0.7) |
| MoE (mergekit-moe) | 0.356 (0.002) | 0.132 (0.002) | 0.278 (0.001) | 21.6 (0.5) |
