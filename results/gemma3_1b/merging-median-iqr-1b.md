# Data-parallel merging by median and interquartile range: Gemma 3 1B

Generated 2026-11-07 by `evaluate/make_merging_report.py` from the per-model JSONs in `results/gemma3_1b/` (as of commit b555f8c).

Contents: data-parallel merging of crime, wine, XSum and DE-EN translation (Gemma 3 1B); whether the summary statistic changes the verdict; outlying seeds.

Each cell is the median over seeds 1-5 with the interquartile range in brackets, *median (IQR)*, as each table's corner says; the base model is evaluated once. Quartiles interpolate linearly between the sorted values (Python's `statistics.quantiles` with `method="inclusive"`, numpy's default), so with five seeds Q1 and Q3 are the second and fourth smallest values and the IQR, Q3 - Q1, is the full width of the middle three seeds, not a ± half-width. For five seeds that scatter normally it averages 0.99 standard deviations, about 1.05 times the sample SD the mean ± SD reports give. That is only 0.73 of the IQR of the distribution the seeds come from (1.35 standard deviations), so these IQRs understate the spread, by the same factor in every five-seed cell, which leaves comparisons between arms unaffected. Q1 and Q3 are single seeds, so the IQR is itself noisy, and (0.000) means only that the second and fourth seeds agree at the precision shown.

One seed that fails can move the median and IQR only as far as a neighbouring seed's value, whereas it can move the mean and standard deviation without limit. The price is that the median and IQR ignore the best and worst seed, so a method that fails on one seed in five can look reliable: cells with such a seed are marked † and listed under *Outlying seeds*.

The data-parallel tables are also in `results/gemma3_1b/{crime,wine,xsum,mt-domain-de-en}-data-parallel-gemma-1b.md` as mean ± SD. Those are built from the `.eval` logs, and these JSONs reproduce every row of them (`--stat mean`), so any difference comes from the statistic alone.

Every arm is repeated with seeds 1-5. The seed sets the training data order (for the LoRA experts also the adapter initialisation and dropout) and mergekit's random seed, which dare_ties uses to choose the weights it drops and arcee_fusion to sample the tensors its thresholds are estimated from.

Merge settings (`merge/make_merge_configs.py`): linear, ties, dare_ties and task_arithmetic give each of the N models merged weight 1/N (N = 2, 4 or 8 in data-parallel merging); ties density 0.5; dare_ties density 0.3; model_stock takes no weights. slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree of pairwise merges, with the first model of each pair as mergekit's base model; arcee_fusion keeps that base and adds the other model's selected changes, so it is not symmetric. With weights summing to 1 on a shared base, linear and task_arithmetic are the same model in exact arithmetic, so any difference between their rows is bf16 rounding and its effect on greedy decoding. At 1B, Axolotl resizes the embedding to the tokenizer (Gemma 3 1B's tokenizer has one token more than its embedding), so PEFT saves embed_tokens and lm_head in each LoRA adapter; `strip_resized_embeddings.py`, run after training, checks that they are the base's plus that row and drops them, and the LoRA merges name no tokenizer source.

## Data-parallel merging: Crime classification (Reddit posts), Gemma 3 1B

The training data is split into N equal parts; an expert, a LoRA adapter, is trained on each part and the N experts are merged. Test set: the 1,077 test posts; accuracy and F1 for the crime class; the main metric is accuracy. *Experts* is the median over the N experts of each seed, then the median over seeds.

### Crime, N = 2

| median (IQR) | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.664 | 0.395 | – |
| **Full data** (one model) | 0.971 (0.005) | 0.968 (0.005) | 5 |
| Experts (each on 1/2 of the data) | 0.951 (0.005) | 0.945 (0.005) | 5 |
| merge: linear | 0.951 (0.004) | 0.944 (0.004) | 5 |
| merge: ties | 0.954 (0.010) | 0.947 (0.012) | 5 |
| merge: dare_ties | 0.952 (0.003) | 0.945 (0.003) | 5 |
| merge: task_arithmetic | 0.950 (0.004) | 0.944 (0.005) | 5 |
| merge: model_stock | 0.931 (0.006) | 0.921 (0.008) | 5 |
| merge: slerp | 0.951 (0.004) | 0.945 (0.004) | 5 |
| merge: arcee_fusion | 0.952 (0.003) | 0.945 (0.002) | 5 |

### Crime, N = 4

| median (IQR) | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.664 | 0.395 | – |
| **Full data** (one model) | 0.971 (0.005) | 0.968 (0.005) | 5 |
| Experts (each on 1/4 of the data) | 0.927 (0.002) | 0.918 (0.004) | 5 |
| merge: linear | 0.933 (0.002) | 0.924 (0.003) | 5 |
| merge: ties | 0.936 (0.014) | 0.929 (0.014) | 5 |
| merge: dare_ties | 0.931 (0.004) | 0.924 (0.003) | 5 |
| merge: task_arithmetic | 0.933 (0.002) | 0.925 (0.002) | 5 |
| merge: model_stock | 0.925 (0.002) | 0.914 (0.002) | 5 |
| merge: slerp | 0.933 (0.006) | 0.925 (0.005) | 5 |
| merge: arcee_fusion | 0.931 (0.004) | 0.922 (0.006) | 5 |

### Crime, N = 8

| median (IQR) | Accuracy | F1 | seeds |
|---|---|---|---|
| Base model (untrained) | 0.664 | 0.395 | – |
| **Full data** (one model) | 0.971 (0.005) | 0.968 (0.005) | 5 |
| Experts (each on 1/8 of the data) | 0.898 (0.002) | 0.885 (0.002) | 5 |
| merge: linear | 0.906 (0.001) | 0.895 (0.000) | 5 |
| merge: ties | 0.916 (0.006) | 0.907 (0.007) | 5 |
| merge: dare_ties | 0.906 (0.001) | 0.894 (0.001) | 5 |
| merge: task_arithmetic | 0.905 (0.003) | 0.894 (0.004) | 5 |
| merge: model_stock | 0.903 (0.003) | 0.889 (0.005) | 5 |
| merge: slerp | 0.905 (0.002) | 0.894 (0.004) | 5 |
| merge: arcee_fusion | 0.907 (0.004) | 0.895 (0.005) | 5 |

## Data-parallel merging: Wine review points (80-100), Gemma 3 1B

The training data is split into N equal parts; an expert, a LoRA adapter, is trained on each part and the N experts are merged. Test set: the 1,000 test reviews; mean absolute and root-mean-square error in points and Pearson correlation over the answered reviews, and the share with no number in the answer; the main metric is MAE (lower is better). *Experts* is the median over the N experts of each seed, then the median over seeds.

### Wine, N = 2

| median (IQR) | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 13.130 | 13.977 | 0.596 | 0.000 | – |
| **Full data** (one model) | 1.302 (0.005) | 1.721 (0.007) | 0.844 (0.002) | 0.000 (0.000) | 5 |
| Experts (each on 1/2 of the data) | 1.374 (0.012) | 1.826 (0.023) | 0.828 (0.001) | 0.000 (0.000) | 5 |
| merge: linear | 1.353 (0.032) | 1.793 (0.038) | 0.832 (0.005) | 0.000 (0.000) | 5 |
| merge: ties | 1.314 (0.027) | 1.725 (0.038) | 0.837 (0.007) | 0.000 (0.000) | 5 |
| merge: dare_ties | 1.370 (0.061) | 1.802 (0.082) | 0.833 (0.005) | 0.000 (0.000) | 5 |
| merge: task_arithmetic | 1.354 (0.034) | 1.791 (0.041) | 0.833 (0.004) | 0.000 (0.000) | 5 |
| merge: model_stock | 1.581 (0.029) | 2.056 (0.028) | 0.815 (0.002) | 0.000 (0.000) | 5 |
| merge: slerp | 1.358 (0.034) | 1.799 (0.036) | 0.832 (0.002) | 0.000 (0.000) | 5 |
| merge: arcee_fusion | 1.349 (0.021) | 1.783 (0.018) | 0.830 (0.001) | 0.000 (0.000) | 5 |

### Wine, N = 4

| median (IQR) | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 13.130 | 13.977 | 0.596 | 0.000 | – |
| **Full data** (one model) | 1.302 (0.005) | 1.721 (0.007) | 0.844 (0.002) | 0.000 (0.000) | 5 |
| Experts (each on 1/4 of the data) | 1.431 (0.012) | 1.898 (0.009) | 0.811 (0.003) | 0.000 (0.000) | 5 |
| merge: linear | 1.412 (0.007) | 1.885 (0.012) | 0.816 (0.004) | 0.000 (0.000) | 5 |
| merge: ties | 1.362 (0.007) | 1.808 (0.009) | 0.822 (0.007) | 0.000 (0.000) | 5 |
| merge: dare_ties | 1.416 (0.013) | 1.892 (0.015) | 0.816 (0.004) | 0.000 (0.000) | 5 |
| merge: task_arithmetic | 1.424 (0.004) | 1.899 (0.006) | 0.817 (0.004) | 0.000 (0.000) | 5 |
| merge: model_stock | 1.540 (0.028) | 2.030 (0.015) | 0.806 (0.003) | 0.000 (0.000) | 5 |
| merge: slerp | 1.425 (0.010) | 1.894 (0.008) | 0.816 (0.004) | 0.000 (0.000) | 5 |
| merge: arcee_fusion | 1.426 (0.012) | 1.901 (0.012) | 0.814 (0.002) | 0.000 (0.000) | 5 |

### Wine, N = 8

| median (IQR) | MAE | RMSE | Pearson | Unanswered | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 13.130 | 13.977 | 0.596 | 0.000 | – |
| **Full data** (one model) | 1.302 (0.005) | 1.721 (0.007) | 0.844 (0.002) | 0.000 (0.000) | 5 |
| Experts (each on 1/8 of the data) | 1.525 (0.006) | 2.028 (0.016) | 0.786 (0.004) | 0.000 (0.000) | 5 |
| merge: linear | 1.510 (0.012) | 2.005 (0.003) | 0.793 (0.003) | 0.000 (0.000) | 5 |
| merge: ties | 1.433 (0.004) | 1.887 (0.017) | 0.801 (0.002) | 0.000 (0.000) | 5 |
| merge: dare_ties | 1.510 (0.023) | 2.006 (0.009) | 0.792 (0.004) | 0.000 (0.000) | 5 |
| merge: task_arithmetic | 1.502 (0.024) | 2.004 (0.010) | 0.792 (0.003) | 0.000 (0.000) | 5 |
| merge: model_stock | 1.543 (0.033) | 2.043 (0.020) | 0.793 (0.006) | 0.000 (0.000) | 5 |
| merge: slerp | 1.517 (0.018) | 2.011 (0.021) | 0.792 (0.002) | 0.000 (0.000) | 5 |
| merge: arcee_fusion | 1.514 (0.008) | 2.001 (0.011) | 0.790 (0.004) | 0.000 (0.000) | 5 |

## Data-parallel merging: XSum summarisation (gold targets, long prompt), Gemma 3 1B

The training data is split into N equal parts; an expert, a student, by self-distillation, is trained on each part and the N experts are merged. Test set: 1,000 test articles, long prompt; ROUGE F-measures, and mean summary length in words; the main metric is ROUGE-1. *Experts* is the median over the N experts of each seed, then the median over seeds.

### XSum, N = 2

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.223 | 0.040 | 0.166 | 19.6 | – |
| **Full data** (one model) | 0.303 (0.002) | 0.088 (0.002) | 0.229 (0.002) | 19.6 (0.3) † | 5 |
| Experts (each on 1/2 of the data) | 0.301 (0.001) | 0.088 (0.001) | 0.230 (0.001) | 19.6 (0.1) | 5 |
| merge: linear | 0.301 (0.004) | 0.089 (0.001) | 0.230 (0.002) | 19.5 (0.3) | 5 |
| merge: ties | 0.303 (0.003) | 0.089 (0.001) | 0.232 (0.001) | 19.3 (0.1) | 5 |
| merge: dare_ties | 0.302 (0.004) | 0.089 (0.001) | 0.230 (0.002) | 19.7 (0.2) | 5 |
| merge: task_arithmetic | 0.302 (0.002) | 0.089 (0.001) | 0.230 (0.001) | 19.4 (0.3) | 5 |
| merge: model_stock | 0.297 (0.002) | 0.083 (0.002) | 0.225 (0.002) | 20.8 (0.1) | 5 |
| merge: slerp | 0.303 (0.003) | 0.089 (0.001) | 0.229 (0.002) | 19.5 (0.2) | 5 |
| merge: arcee_fusion | 0.303 (0.002) | 0.089 (0.001) | 0.230 (0.000) | 19.6 (0.2) | 5 |

### XSum, N = 4

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.223 | 0.040 | 0.166 | 19.6 | – |
| **Full data** (one model) | 0.303 (0.002) | 0.088 (0.002) | 0.229 (0.002) | 19.6 (0.3) † | 5 |
| Experts (each on 1/4 of the data) | 0.298 (0.003) | 0.086 (0.001) | 0.228 (0.002) | 18.8 (0.1) | 5 |
| merge: linear | 0.298 (0.003) | 0.088 (0.002) | 0.229 (0.002) | 18.8 (0.3) | 5 |
| merge: ties | 0.298 (0.003) | 0.091 (0.000) | 0.231 (0.003) | 18.4 (0.2) | 5 |
| merge: dare_ties | 0.296 (0.003) | 0.087 (0.001) | 0.228 (0.003) | 18.8 (0.2) | 5 |
| merge: task_arithmetic | 0.297 (0.002) | 0.088 (0.001) | 0.228 (0.001) | 18.6 (0.3) | 5 |
| merge: model_stock | 0.295 (0.003) | 0.084 (0.001) | 0.225 (0.001) | 19.4 (0.4) | 5 |
| merge: slerp | 0.296 (0.001) | 0.088 (0.002) | 0.228 (0.001) | 18.8 (0.3) | 5 |
| merge: arcee_fusion | 0.296 (0.002) | 0.088 (0.000) | 0.228 (0.001) | 18.9 (0.2) | 5 |

### XSum, N = 8

| median (IQR) | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 0.223 | 0.040 | 0.166 | 19.6 | – |
| **Full data** (one model) | 0.303 (0.002) | 0.088 (0.002) | 0.229 (0.002) | 19.6 (0.3) † | 5 |
| Experts (each on 1/8 of the data) | 0.287 (0.002) | 0.079 (0.001) | 0.219 (0.000) | 18.9 (0.1) | 5 |
| merge: linear | 0.285 (0.001) | 0.079 (0.002) | 0.218 (0.000) | 18.8 (0.3) | 5 |
| merge: ties | 0.288 (0.000) | 0.081 (0.001) | 0.221 (0.000) | 18.9 (0.2) | 5 |
| merge: dare_ties | 0.285 (0.001) | 0.080 (0.001) | 0.218 (0.000) | 18.8 (0.3) | 5 |
| merge: task_arithmetic | 0.284 (0.001) | 0.079 (0.002) | 0.218 (0.002) | 18.8 (0.3) | 5 |
| merge: model_stock | 0.285 (0.002) | 0.078 (0.002) | 0.217 (0.002) | 19.0 (0.2) | 5 |
| merge: slerp | 0.284 (0.002) | 0.078 (0.000) | 0.218 (0.001) | 18.8 (0.3) | 5 |
| merge: arcee_fusion | 0.284 (0.001) | 0.079 (0.001) | 0.218 (0.001) | 18.8 (0.4) | 5 |

## Data-parallel merging: German-English translation (medical and legal), Gemma 3 1B

The training data is split into N equal parts; an expert, a LoRA adapter, is trained on each part and the N experts are merged. Test set: the 2,000 EMEA and JRC-Acquis test pairs (1,000 per domain); mean sentence chrF++, overall and per domain, and output length over reference length in words; the main metric is chrF++. *Experts* is the median over the N experts of each seed, then the median over seeds.

### DE-EN, N = 2

| median (IQR) | chrF++ | Medical | Legal | Length ratio | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 32.41 | 33.26 | 31.57 | 6.76 | – |
| **Full data** (one model) | 57.85 (0.14) | 59.36 (0.20) | 56.35 (0.08) | 1.12 (0.11) | 5 |
| Experts (each on 1/2 of the data) | 55.84 (0.11) | 57.56 (0.16) | 54.13 (0.15) | 1.09 (0.05) | 5 |
| merge: linear | 56.53 (0.25) | 58.10 (0.16) | 54.75 (0.33) | 1.04 (0.03) | 5 |
| merge: ties | 55.53 (0.57) | 57.34 (0.29) | 53.98 (0.43) | 1.09 (0.04) | 5 |
| merge: dare_ties | 56.26 (0.30) | 57.99 (0.27) | 54.65 (0.38) | 1.05 (0.02) | 5 |
| merge: task_arithmetic | 56.45 (0.19) | 58.26 (0.33) | 54.75 (0.15) | 1.05 (0.05) | 5 |
| merge: model_stock | 55.37 (0.15) | 57.02 (0.09) | 53.75 (0.21) | 1.03 (0.01) | 5 |
| merge: slerp | 56.44 (0.01) | 58.18 (0.35) | 54.69 (0.27) | 1.04 (0.02) | 5 |
| merge: arcee_fusion | 56.12 (0.06) † | 57.95 (0.33) | 54.32 (0.26) | 1.06 (0.02) | 5 |

### DE-EN, N = 4

| median (IQR) | chrF++ | Medical | Legal | Length ratio | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 32.41 | 33.26 | 31.57 | 6.76 | – |
| **Full data** (one model) | 57.85 (0.14) | 59.36 (0.20) | 56.35 (0.08) | 1.12 (0.11) | 5 |
| Experts (each on 1/4 of the data) | 54.07 (0.10) | 55.77 (0.11) | 52.39 (0.04) | 1.13 (0.02) | 5 |
| merge: linear | 54.65 (0.06) | 56.30 (0.14) | 53.04 (0.21) | 1.07 (0.13) | 5 |
| merge: ties | 54.06 (0.44) | 55.92 (0.60) | 52.02 (0.27) | 1.11 (0.09) | 5 |
| merge: dare_ties | 54.57 (0.15) | 56.14 (0.12) | 52.88 (0.19) | 1.07 (0.16) | 5 |
| merge: task_arithmetic | 54.61 (0.27) | 56.31 (0.08) | 52.98 (0.34) | 1.07 (0.14) | 5 |
| merge: model_stock | 54.50 (0.04) | 56.13 (0.40) | 52.99 (0.44) | 1.03 (0.02) | 5 |
| merge: slerp | 54.70 (0.10) | 56.34 (0.16) | 53.09 (0.23) | 1.07 (0.17) | 5 |
| merge: arcee_fusion | 54.63 (0.14) | 56.33 (0.45) | 52.81 (0.19) | 1.22 (0.12) | 5 |

### DE-EN, N = 8

| median (IQR) | chrF++ | Medical | Legal | Length ratio | seeds |
|---|---|---|---|---|---|
| Base model (untrained) | 32.41 | 33.26 | 31.57 | 6.76 | – |
| **Full data** (one model) | 57.85 (0.14) | 59.36 (0.20) | 56.35 (0.08) | 1.12 (0.11) | 5 |
| Experts (each on 1/8 of the data) | 52.35 (0.25) | 53.95 (0.11) | 50.76 (0.24) | 1.20 (0.04) | 5 |
| merge: linear | 52.77 (0.19) | 54.42 (0.08) | 51.12 (0.07) | 1.20 (0.10) | 5 |
| merge: ties | 51.91 (0.30) | 53.62 (0.16) | 50.05 (0.22) | 1.36 (0.23) | 5 |
| merge: dare_ties | 52.70 (0.10) | 54.37 (0.13) | 51.06 (0.24) | 1.15 (0.14) | 5 |
| merge: task_arithmetic | 52.78 (0.06) | 54.49 (0.01) | 51.03 (0.19) | 1.16 (0.07) | 5 |
| merge: model_stock | 53.01 (0.09) | 54.60 (0.16) | 51.43 (0.02) | 1.07 (0.05) | 5 |
| merge: slerp | 52.76 (0.00) | 54.47 (0.12) | 51.06 (0.12) | 1.21 (0.13) | 5 |
| merge: arcee_fusion | 52.58 (0.34) | 54.22 (0.20) | 50.92 (0.30) | 1.14 (0.07) | 5 |

## Does the summary statistic change the verdict?

In 11 of the 12 comparisons the best merge by the median is also the best by the mean, ties at the precision shown counting as best; it changes in Data-parallel DE-EN, N = 2. The worst merge differs between comparisons (last column).

The main metric of each comparison is accuracy (crime), MAE (wine, lower is better), ROUGE-1 (XSum) and chrF++ (DE-EN translation). Methods are tied when they agree at the precision shown, and tied methods are listed together, the better unrounded first. *Next* is the methods with the second-best median. *Wins* counts the seeds on which the first best method beats each of them, out of the seeds both have: every method of a seed merges the same experts, so the comparison is paired. With five seeds neither the medians nor the win counts are a significance test; they show how consistent a lead is. *Same best* says whether the best by the median and by the mean have a method in common.

| Comparison | Best by median | Next by median | Wins | Best by mean | Same best | Worst by median |
|---|---|---|---|---|---|---|
| Data-parallel Crime, N = 2 | ties 0.954 | dare_ties = arcee_fusion 0.952 | 4 of 5, 4 of 5 | ties 0.955 | yes | model_stock |
| Data-parallel Crime, N = 4 | ties 0.936 | linear = task_arithmetic = slerp 0.933 | 3 of 5, 3 of 5, 3 of 5 | linear = dare_ties = ties = task_arithmetic = slerp 0.933 | yes | model_stock |
| Data-parallel Crime, N = 8 | ties 0.916 | arcee_fusion 0.907 | 4 of 5 | ties 0.914 | yes | model_stock |
| Data-parallel Wine, N = 2 | ties 1.314 | arcee_fusion 1.349 | 5 of 5 | ties 1.314 | yes | model_stock |
| Data-parallel Wine, N = 4 | ties 1.362 | linear 1.412 | 5 of 5 | ties 1.363 | yes | model_stock |
| Data-parallel Wine, N = 8 | ties 1.433 | task_arithmetic 1.502 | 5 of 5 | ties 1.430 | yes | model_stock |
| Data-parallel XSum, N = 2 | arcee_fusion = ties = slerp 0.303 | dare_ties = task_arithmetic 0.302 | 3 of 5, 3 of 5 | ties 0.304 | yes | model_stock |
| Data-parallel XSum, N = 4 | ties = linear 0.298 | task_arithmetic 0.297 | 4 of 5 | ties 0.299 | yes | model_stock |
| Data-parallel XSum, N = 8 | ties 0.288 | model_stock = linear = dare_ties 0.285 | 5 of 5, 5 of 5, 5 of 5 | ties 0.288 | yes | arcee_fusion = slerp = task_arithmetic |
| Data-parallel DE-EN, N = 2 | linear 56.53 | task_arithmetic 56.45 | 2 of 5 | task_arithmetic 56.45 | no | model_stock |
| Data-parallel DE-EN, N = 4 | slerp 54.70 | linear 54.65 | 3 of 5 | slerp 54.66 | yes | ties |
| Data-parallel DE-EN, N = 8 | model_stock 53.01 | task_arithmetic 52.78 | 5 of 5 | model_stock 53.03 | yes | ties |

## Outlying seeds

Cells in which a seed lies more than 3 IQRs beyond the nearer quartile, taking as the IQR the larger of the cell's own and the median IQR of that metric over every arm of the study (with five seeds a cell's own IQR is often a unit or two of the last digit, which would make ordinary noise outlying): 2 of the 350 distinct five-seed cells. They are marked † in the tables. The median and IQR barely move for such a seed, which is the point of them, but they also hide it, and a method that fails on one seed in five is less reliable than its median suggests. The outlying values are in bold; *Mean ± SD* is computed as in the mean ± SD reports.

| Where | Row | Metric | Seeds 1-5 | Mean ± SD | Median (IQR) |
|---|---|---|---|---|---|
| Data-parallel XSum | **Full data** (one model) | Words | 19.6, 19.4, 19.6, 19.2, **20.5** | 19.6 ± 0.5 | 19.6 (0.3) |
| Data-parallel DE-EN, N = 2 | merge: arcee_fusion | chrF++ | 56.12, 56.10, **56.60**, 56.06, 56.16 | 56.21 ± 0.22 | 56.12 (0.06) |

By arm and seed, with the arm's score on that seed and on its other four, on the main metric of its comparison (as in the verdict above), so that a seed can be judged as a whole: a seed outlying on one metric may still be weak, or fine, overall.

| Where | Row | Seed | Outlying on | Score on that seed | Other seeds |
|---|---|---|---|---|---|
| Data-parallel XSum | **Full data** (one model) | 5 | Words | ROUGE-1 0.305 | 0.299 to 0.304 |
| Data-parallel DE-EN, N = 2 | merge: arcee_fusion | 3 | chrF++ | chrF++ 56.60 | 56.06 to 56.16 |
