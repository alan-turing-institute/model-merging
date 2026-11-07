# Routing the task MoE: Mistral 7B, seeds 1-7

Main's sparse task MoE (`merge/moe/mistral_task_moe_4_seed_<S>_config.yaml`: four MLP-only experts, crime, wine, CLINC150 with the labels prompt and XSum by self-distillation, one expert per token, mergekit's `gate_mode: hidden`) keeps most of its experts' gains on wine and CLINC150 but loses much of crime's. This study rebuilt it and asked whether a better router recovers crime. The experts and the MoE were rebuilt from main (7122af6) with its configs, because main's models are not readable to us; every router below routes the same four experts of a seed, and is fitted on training data only. Each MoE is evaluated as main evaluates its MoEs (test sets in full; CLINC150 with the labels prompt, XSum with the long one).

Seeds 1 and 2 are complete. The later seeds were cut short when the project's GPU budget on Isambard ran out (6 November 2026), so they cover: main's router: seeds 1-6 (some tasks also for 7); router 1: seeds 1-4 (some tasks also for 5); router 2: seeds 1-3; router 3: seeds 1 and 2.

*R* is the share of the expert's gain over the base model a model keeps, for one task and seed: (model - base) / (expert of the same seed - base), on the task's main metric (wine's MAE is lower-better, which the ratio handles). 1 is as good as the expert, 0 no better than the base model. *All four* is the mean of a seed's four R values.

## The rebuild

Main's MoE (its own experts) and the rebuild (ours), same config and code:

| | seed | Crime accuracy | Wine MAE | CLINC150 accuracy | XSum ROUGE-1 |
|---|---|---|---|---|---|
| main's MoE | 1 | 0.876 | 1.210 | 0.973 | 0.363 |
| main's MoE | 2 | 0.906 | 1.223 | 0.975 | 0.332 |
| main's MoE | 3 | 0.920 | 1.187 | 0.976 | 0.351 |
| main's MoE | 4 | 0.941 | 1.188 | 0.972 | 0.351 |
| main's MoE | 5 | 0.835 | 1.194 | 0.978 | 0.281 |
| rebuilt MoE | 1 | 0.848 | 1.218 | 0.975 | 0.361 |
| rebuilt MoE | 2 | 0.896 | 1.212 | 0.978 | 0.331 |
| rebuilt MoE | 3 | 0.911 | 1.201 | 0.976 | 0.348 |
| rebuilt MoE | 4 | 0.922 | 1.195 | 0.973 | 0.354 |
| rebuilt MoE | 5 | 0.783 | 1.189 | 0.977 | 0.280 |
| rebuilt MoE | 6 | 0.898 | 1.199 | 0.977 | 0.222 |
| rebuilt MoE | 7 | 0.916 | 1.224 | 0.978 | – |

The rebuild's crime accuracy (0.783-0.922 over 7 seeds) overlaps main's (0.835-0.941 over 5), so it reproduces the weakness rather than an accident of training.

## Where crime's routing fails

The probe (`merge/moe_routing.py probe`, 300 test prompts per task) records the expert each layer picks for every prompt token. A crime answer is decided at the answer position, the last prompt token. Share of routing decisions there going to each expert, over all 32 layers, and the share going home in layers 0-7; for crime also split by whether the MoE answered correctly:

| router | seed | to crime | to wine | to CLINC150 | to XSum | home, layers 0-7 | home when right (n) | home when wrong (n) |
|---|---|---|---|---|---|---|---|---|
| moe | 1 | 0.47 | 0.26 | 0.25 | 0.01 | 0.04 | 0.51 (251) | 0.27 (49) |
| moe | 2 | 0.40 | 0.28 | 0.30 | 0.01 | 0.03 | 0.41 (266) | 0.34 (34) |
| moe_prompts | 1 | 0.93 | 0.00 | 0.01 | 0.07 | 0.73 | 0.93 (296) | 0.91 (4) |
| moe_prompts | 2 | 0.91 | 0.00 | 0.03 | 0.05 | 0.79 | 0.92 (293) | 0.84 (7) |
| moe_fit | 1 | 0.91 | 0.01 | 0.02 | 0.06 | 1.00 | 0.91 (296) | 0.99 (4) |
| moe_fit | 2 | 0.88 | 0.01 | 0.02 | 0.09 | 1.00 | 0.88 (296) | 0.95 (4) |
| moe_train | 1 | 0.99 | 0.00 | 0.00 | 0.00 | 0.98 | 0.99 (296) | 0.93 (4) |
| moe_train | 2 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 (296) | 1.00 (4) |

With main's router, about half of the decisions at crime's answer position go to the wine and CLINC150 experts, nearly all of them in the early layers, and the wrong answers are routed home less often than the right ones. mergekit's hidden router is built from the base model's state at the last token of each positive prompt, and main's prompts are bare instructions and post titles without the chat template, so no expert's router row stands for the answer position. Main found the same independently (fb26f01).

## Routers

1. **Better prompts** (`moe_routing.py prompts`): mergekit as before, but each expert's positive prompts are 32 real training examples rendered with the chat template, 16 ending at the answer position and 16 cut at a random token (`merge/moe/mistral_task_moe_4_seed_<S>_prompts_config.yaml`).
2. **Fitted** (`moe_routing.py fit`): each layer's router is a multinomial logistic regression, without bias like the router, of the task on the base model's router input (post-attention, normalised hidden state), over 300 training sequences per task (prompt and answer), 20,000 tokens per task and layer, the answer position always included.
3. **Trained** (`moe_routing.py train`): the routers alone (32 x 4 x 4096 = 524,288 parameters; everything else frozen) trained on the MoE itself for 600 steps, tasks in turn, from main's router: a per-token routing loss towards the sequence's task, plus the answer tokens' loss. With one expert per token the chosen expert's weight is renormalised to 1, so the answer loss reaches the router only through a straight-through weight (top / top.detach(), 1 in the forward pass).
4. **Per prompt**: route every token of a prompt to one expert, chosen once per prompt. An MoE that sends every token to expert e is expert e's model, since the experts differ from the base only in their MLPs (main checked this for its MoEs: argmax agreement 0.97-1.00), so this router scores as the expert on every prompt it assigns correctly; what remains to measure is that assignment (below).

## Results

| model | seed | Crime accuracy | Wine MAE | CLINC150 accuracy | XSum ROUGE-1 | R crime | R wine | R CLINC150 | R XSum | All four |
|---|---|---|---|---|---|---|---|---|---|---|
| base model | – | 0.776 | 1.685 | 0.454 | 0.237 | | | | | |
| experts | 1 | 0.982 | 1.137 | 0.975 | 0.378 | 1 | 1 | 1 | 1 | 1 |
| experts | 2 | 0.983 | 1.140 | 0.979 | 0.375 | 1 | 1 | 1 | 1 | 1 |
| experts | 3 | 0.985 | 1.129 | 0.977 | 0.374 | 1 | 1 | 1 | 1 | 1 |
| experts | 4 | 0.983 | 1.127 | 0.978 | 0.372 | 1 | 1 | 1 | 1 | 1 |
| experts | 5 | 0.980 | 1.153 | 0.977 | 0.374 | 1 | 1 | 1 | 1 | 1 |
| experts | 6 | 0.981 | 1.129 | 0.978 | 0.379 | 1 | 1 | 1 | 1 | 1 |
| experts | 7 | 0.981 | 1.142 | 0.978 | 0.375 | 1 | 1 | 1 | 1 | 1 |
| main's router (mergekit hidden, main's prompts) | 1 | 0.848 | 1.218 | 0.975 | 0.361 | 0.35 | 0.85 | 1.00 | 0.88 | 0.77 |
| main's router (mergekit hidden, main's prompts) | 2 | 0.896 | 1.212 | 0.978 | 0.331 | 0.58 | 0.87 | 1.00 | 0.68 | 0.78 |
| main's router (mergekit hidden, main's prompts) | 3 | 0.911 | 1.201 | 0.976 | 0.348 | 0.64 | 0.87 | 1.00 | 0.81 | 0.83 |
| main's router (mergekit hidden, main's prompts) | 4 | 0.922 | 1.195 | 0.973 | 0.354 | 0.70 | 0.88 | 0.99 | 0.87 | 0.86 |
| main's router (mergekit hidden, main's prompts) | 5 | 0.783 | 1.189 | 0.977 | 0.280 | 0.03 | 0.93 | 1.00 | 0.31 | 0.57 |
| main's router (mergekit hidden, main's prompts) | 6 | 0.898 | 1.199 | 0.977 | 0.222 | 0.60 | 0.87 | 1.00 | -0.10 | 0.59 |
| main's router (mergekit hidden, main's prompts) | 7 | 0.916 | 1.224 | 0.978 | – | 0.69 | 0.85 | 1.00 | – | – |
| 1. real chat-formatted examples as mergekit's prompts | 1 | 0.980 | 1.142 | 0.973 | 0.369 | 0.99 | 0.99 | 1.00 | 0.94 | 0.98 |
| 1. real chat-formatted examples as mergekit's prompts | 2 | 0.976 | 1.147 | 0.976 | 0.370 | 0.96 | 0.99 | 0.99 | 0.96 | 0.98 |
| 1. real chat-formatted examples as mergekit's prompts | 3 | 0.961 | 1.136 | 0.977 | 0.371 | 0.88 | 0.99 | 1.00 | 0.97 | 0.96 |
| 1. real chat-formatted examples as mergekit's prompts | 4 | 0.977 | 1.134 | 0.971 | 0.371 | 0.97 | 0.99 | 0.99 | 1.00 | 0.99 |
| 1. real chat-formatted examples as mergekit's prompts | 5 | – | – | – | 0.368 | – | – | – | 0.96 | – |
| 2. fitted on hidden states (logistic regression) | 1 | 0.982 | 1.151 | 0.975 | 0.374 | 1.00 | 0.97 | 1.00 | 0.97 | 0.99 |
| 2. fitted on hidden states (logistic regression) | 2 | 0.983 | 1.145 | 0.977 | 0.371 | 1.00 | 0.99 | 1.00 | 0.97 | 0.99 |
| 2. fitted on hidden states (logistic regression) | 3 | 0.985 | 1.134 | 0.976 | 0.376 | 1.00 | 0.99 | 1.00 | 1.01 | 1.00 |
| 3. trained (routing + answer loss) | 1 | 0.982 | 1.143 | 0.975 | 0.377 | 1.00 | 0.99 | 1.00 | 0.99 | 1.00 |
| 3. trained (routing + answer loss) | 2 | 0.984 | 1.156 | 0.980 | 0.374 | 1.00 | 0.97 | 1.00 | 0.99 | 0.99 |

Over seeds: R on crime for every seed with a crime evaluation, *All four* for every seed with all four tasks; median (range):

| router | seeds, crime | R crime | seeds, all four | All four |
|---|---|---|---|---|
| main's router (mergekit hidden, main's prompts) | 1-7 | 0.60 (0.03 to 0.70) | 1-6 | 0.78 (0.57 to 0.86) |
| 1. real chat-formatted examples as mergekit's prompts | 1-4 | 0.97 (0.88 to 0.99) | 1-4 | 0.98 (0.96 to 0.99) |
| 2. fitted on hidden states (logistic regression) | 1-3 | 1.00 (1.00 to 1.00) | 1-3 | 0.99 (0.99 to 1.00) |
| 3. trained (routing + answer loss) | 1 and 2 | 1.00 (1.00 to 1.00) | 1 and 2 | 0.99 (0.99 to 1.00) |

## Routing per prompt

Share of each task's test prompts whose mean router probability (over tokens and layers) picks the task's own expert, i.e. how often routing per prompt would use the right expert:

| router | seed | crime | wine | CLINC150 | XSum |
|---|---|---|---|---|---|
| moe | 1 | 0.58 | 1.00 | 1.00 | 0.91 |
| moe | 2 | 0.80 | 1.00 | 1.00 | 0.89 |
| moe_prompts | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| moe_prompts | 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| moe_fit | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| moe_fit | 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| moe_train | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| moe_train | 2 | 1.00 | 1.00 | 1.00 | 1.00 |

In seeds 1 and 2, with any of the three new routers every probed prompt is assigned to its own expert, so routing per prompt would score as the experts do (R = 1). With main's router it would misroute 20-42% of crime prompts.

## Caveats

- Only seeds 1 and 2 are complete and probed; the later seeds lack some routers (above). One model (Mistral 7B); main's MoE also has Llama 3.1 8B.
- The probe uses 300 test prompts per task; the evaluations use the full test sets.
- XSum for router 1 from seed 2 on and router 2 from seed 3 on ran at batch size 1 (the MoE ran out of GPU memory at 2); decoding is greedy, so only the padding differs.
- Routers 2 and 3 were fitted and trained on the training splits only; router 3's progress was checked on 40 validation sequences per task, never on test data.

## Reproducing

On Isambard, in a worktree of main: `merge/submit_moe_rebuild.sh [seed...]` (experts, conversions, base and expert evaluations, main's `submit_moe.sh`), then `merge/submit_routing.sh [seed...]` (probes, the three routers and their evaluations); `evaluate/export_moe_routing.py` writes the JSONs and this script the report.
