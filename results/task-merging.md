# Task merging: crime, wine, CLINC150 and XSum experts merged into one model

Generated 2026-11-05 by `evaluate/make_task_merge_tables.py` from the JSON results in `results/task_merging/` (`index.tsv` there maps each file to its `.eval` log in `evaluate/logs/`).

For each base model, one expert is trained per task, with LoRA on the MLP layers only (crime, wine, CLINC150) or full self-distillation of the MLP layers on gold targets (XSum); the four experts are then merged into one model. Every arm is repeated with seeds 1-5 (expert training, and dare_ties' random dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated once.

Test sets and metrics: crime, 1,077 Reddit posts, accuracy; wine, 1,000 Wine Enthusiast reviews, mean absolute error of the predicted points (lower is better); CLINC150, 4,500 queries with the labels prompt, accuracy; XSum, 1,000 BBC articles with the long prompt, ROUGE-1 F-measure against the gold summary. Greedy decoding throughout.

Each expert is evaluated on all four tasks: its own (the diagonal) and the three it was not trained on (`evaluate/submit_expert_cross_evals.sh`, run 5 Nov). An expert's retention is how much of the four experts' combined gain over the base model it alone gives, so about 0.25 plus whatever it transfers. **Retention** is, for each seed, the mean over the four tasks of (merged − base) / (expert − base), using that seed's expert for each task: 1 means the merge matches every expert on its own task, 0 means no better than the base model, and negative means worse than the base. It is then averaged over seeds.

**†** Wine MAE is over the answers that contain a number (`evaluate/wine_task.py`). A wine cell is marked † (with the share in brackets) when over 2% of answers, averaged over seeds, had no number or a number outside 80-100. This happens to experts off their own task: they often answer with the vintage year ("2017", an error of ~1,930 points) or with text ("not yet rated"), so the MAE, its spread and that row's retention are driven by a few absurd answers and say little about skill at rating wine.

Merge settings (`merge/make_task_merge_configs.py`, the same as for data-parallel merging): weight 1/4 for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a binary tree: (crime, wine) and (CLINC150, XSum), then the two results. The MoE (`merge/make_moe_configs.py`, Mistral and Llama only, since mergekit-moe has no Gemma 3 path) keeps each expert's MLP as one Mixtral expert on the untrained base, with `gate_mode: hidden` routing from task prompts and one expert per token; its XSum evaluations ran at batch size 2 rather than 8 (out of GPU memory), apart from Mistral and Llama seed 3.

## Gemma 3 4B (google/gemma-3-4b-it)

| | Crime acc ↑ | Wine MAE ↓ | CLINC150 acc ↑ | XSum ROUGE-1 ↑ | Retention | seeds |
|---|---|---|---|---|---|---|
| Base model (untrained) | 0.755 | 2.38 † (7%) | 0.704 | 0.247 | 0 | – |
| Expert: crime | 0.980 ± 0.001 | 2.30 ± 0.08 † (4%) | 0.721 ± 0.017 | 0.247 ± 0.003 | 0.28 ± 0.01 | 5 |
| Expert: wine | 0.759 ± 0.021 | 1.18 ± 0.02 | 0.693 ± 0.018 | 0.247 ± 0.005 | 0.25 ± 0.03 | 5 |
| Expert: CLINC150 | 0.688 ± 0.047 | 23.20 ± 28.90 † (3%) | 0.978 ± 0.001 | 0.230 ± 0.012 | -4.21 ± 6.07 | 5 |
| Expert: XSum | 0.732 ± 0.017 | 2.51 ± 0.06 † (6%) | 0.704 ± 0.007 | 0.362 ± 0.003 | 0.20 ± 0.03 | 5 |
| merge: linear | 0.894 ± 0.005 | 1.42 ± 0.04 | 0.959 ± 0.003 | 0.304 ± 0.007 | 0.71 ± 0.02 | 5 |
| merge: ties | 0.942 ± 0.009 | 1.61 ± 0.87 | 0.978 ± 0.001 | 0.260 ± 0.012 | 0.64 ± 0.19 | 5 |
| merge: dare_ties | 0.895 ± 0.006 | 1.41 ± 0.03 | 0.962 ± 0.002 | 0.302 ± 0.006 | 0.71 ± 0.01 | 5 |
| merge: task_arithmetic | 0.894 ± 0.004 | 1.41 ± 0.04 | 0.960 ± 0.003 | 0.304 ± 0.007 | 0.71 ± 0.02 | 5 |
| merge: model_stock | 0.760 ± 0.002 | 2.38 ± 0.02 † (7%) | 0.717 ± 0.000 | 0.247 ± 0.001 | 0.02 ± 0.00 | 5 |
| merge: slerp | 0.895 ± 0.004 | 1.42 ± 0.04 | 0.959 ± 0.003 | 0.304 ± 0.006 | 0.71 ± 0.02 | 5 |
| merge: arcee_fusion | 0.924 ± 0.010 | 1.66 ± 0.07 | 0.935 ± 0.003 | 0.271 ± 0.004 | 0.60 ± 0.01 | 5 |

## Mistral 7B (mistralai/Mistral-7B-Instruct-v0.3)

| | Crime acc ↑ | Wine MAE ↓ | CLINC150 acc ↑ | XSum ROUGE-1 ↑ | Retention | seeds |
|---|---|---|---|---|---|---|
| Base model (untrained) | 0.774 | 1.68 | 0.454 | 0.238 | 0 | – |
| Expert: crime | 0.983 ± 0.002 | 195.17 ± 428.29 † (53%) | 0.030 ± 0.029 | 0.038 ± 0.028 | -87.98 ± 193.97 | 5 |
| Expert: wine | 0.669 ± 0.039 | 1.14 ± 0.01 | 0.315 ± 0.077 | 0.252 ± 0.012 | 0.08 ± 0.08 | 5 |
| Expert: CLINC150 | 0.090 ± 0.062 | 18.31 ± 22.04 † (21%) | 0.975 ± 0.001 | 0.059 ± 0.025 | -8.66 ± 10.34 | 5 |
| Expert: XSum | 0.702 ± 0.009 | 1.60 ± 0.04 | 0.337 ± 0.037 | 0.376 ± 0.002 | 0.14 ± 0.03 | 5 |
| merge: linear | 0.936 ± 0.009 | 1.27 ± 0.01 | 0.973 ± 0.002 | 0.324 ± 0.006 | 0.79 ± 0.02 | 5 |
| merge: ties | 0.954 ± 0.013 | 1.18 ± 0.02 | 0.978 ± 0.002 | 0.050 ± 0.021 | 0.36 ± 0.05 | 5 |
| merge: dare_ties | 0.935 ± 0.010 | 1.27 ± 0.01 | 0.972 ± 0.001 | 0.310 ± 0.007 | 0.76 ± 0.02 | 5 |
| merge: task_arithmetic | 0.936 ± 0.009 | 1.26 ± 0.01 | 0.973 ± 0.002 | 0.324 ± 0.006 | 0.79 ± 0.02 | 5 |
| merge: model_stock | 0.780 ± 0.002 | 1.66 ± 0.01 | 0.472 ± 0.002 | 0.237 ± 0.001 | 0.02 ± 0.01 | 5 |
| merge: slerp | 0.936 ± 0.009 | 1.26 ± 0.01 | 0.973 ± 0.002 | 0.324 ± 0.005 | 0.79 ± 0.02 | 5 |
| merge: arcee_fusion | 0.979 ± 0.004 | 1.33 ± 0.03 | 0.963 ± 0.003 | 0.165 ± 0.048 | 0.52 ± 0.09 | 5 |
| MoE (Mixtral, 4 experts, top-1) | 0.895 ± 0.041 | 1.20 ± 0.02 | 0.975 ± 0.002 | 0.336 ± 0.032 | 0.79 ± 0.10 | 5 |

## Llama 3.1 8B (meta-llama/Llama-3.1-8B-Instruct)

| | Crime acc ↑ | Wine MAE ↓ | CLINC150 acc ↑ | XSum ROUGE-1 ↑ | Retention | seeds |
|---|---|---|---|---|---|---|
| Base model (untrained) | 0.743 | 4.36 | 0.640 | 0.289 | 0 | – |
| Expert: crime | 0.984 ± 0.002 | 2.62 ± 0.27 | 0.617 ± 0.075 | 0.304 ± 0.005 | 0.41 ± 0.08 | 5 |
| Expert: wine | 0.763 ± 0.009 | 1.15 ± 0.02 | 0.731 ± 0.016 | 0.311 ± 0.004 | 0.40 ± 0.02 | 5 |
| Expert: CLINC150 | 0.703 ± 0.049 | 2.50 ± 0.24 | 0.979 ± 0.002 | 0.261 ± 0.013 | 0.27 ± 0.09 | 5 |
| Expert: XSum | 0.716 ± 0.005 | 2.10 ± 0.03 | 0.676 ± 0.003 | 0.379 ± 0.001 | 0.42 ± 0.01 | 5 |
| merge: linear | 0.900 ± 0.002 | 1.25 ± 0.01 | 0.964 ± 0.004 | 0.337 ± 0.002 | 0.78 ± 0.01 | 5 |
| merge: ties | 0.947 ± 0.004 | 1.18 ± 0.01 | 0.979 ± 0.001 | 0.268 ± 0.005 | 0.65 ± 0.02 | 5 |
| merge: dare_ties | 0.901 ± 0.003 | 1.25 ± 0.02 | 0.966 ± 0.004 | 0.334 ± 0.002 | 0.77 ± 0.01 | 5 |
| merge: task_arithmetic | 0.900 ± 0.002 | 1.24 ± 0.01 | 0.964 ± 0.004 | 0.338 ± 0.003 | 0.78 ± 0.01 | 5 |
| merge: model_stock | 0.750 ± 0.002 | 4.14 ± 0.06 | 0.652 ± 0.002 | 0.289 ± 0.001 | 0.03 ± 0.01 | 5 |
| merge: slerp | 0.899 ± 0.003 | 1.25 ± 0.01 | 0.964 ± 0.004 | 0.337 ± 0.002 | 0.78 ± 0.01 | 5 |
| merge: arcee_fusion | 0.959 ± 0.010 | 1.36 ± 0.02 | 0.949 ± 0.003 | 0.309 ± 0.004 | 0.74 ± 0.02 | 5 |
| MoE (Mixtral, 4 experts, top-1) | 0.929 ± 0.007 | 1.18 ± 0.02 | 0.977 ± 0.000 | 0.357 ± 0.002 | 0.88 ± 0.01 | 5 |
