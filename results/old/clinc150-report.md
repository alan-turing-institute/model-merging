# CLINC150: merging, task-parallel merging and distillation

Intent classification on CLINC150: 150 intents in 10 author-defined domains (domain list from
clinc/oos-eval, CC BY 3.0, `train/clinc150_domains.json`). The domains are split five and five
a priori; expert A is trained on domains 1-5, expert B on domains 6-10. Out-of-scope queries are
dropped, since each expert would call the other's domains out of scope. Gemma 3 4B and 1B, LoRA;
accuracy on the 4,500 test queries with the `short` prompt, which does not list the labels.
Work of 10-14 October 2026.

These results come from the screen pipeline on branch `samsum-scaling` (4-bit QLoRA; the code
under Sources, which is not on main). `train/prepare_data_clinc150.py` here builds the same
split by the same procedure (same domain shards, prompt, shuffle seed and test set), with the
shards named 1 and 2 rather than A and B; rerunning the results with main's pipeline (bf16 LoRA,
`train.sbatch`, `make_merge_configs.py`, `evaluate.sbatch clinc150`) is still to do.

## 1. Headroom screen (4B)

| arm | accuracy | domains 1-5 | domains 6-10 |
|---|---|---|---|
| untrained, short prompt | 0.051 | 0.045 | 0.057 |
| untrained, all 150 labels in the prompt | 0.704 | 0.692 | 0.716 |
| full data | 0.977 | 0.979 | 0.975 |
| random half (matched size) | 0.969 | 0.973 | 0.965 |
| expert A | 0.547 | 0.977 | 0.117 |
| expert B | 0.543 | 0.109 | 0.977 |

A random half falls 0.008 [0.004, 0.012] short of the full model, too little to measure merging
against. The domain experts fall 0.430 [0.414, 0.444] short, 86 times the assumed seed spread:
each matches the full model on its own domains and scores 0.11-0.12 on the other's. Listing the
labels lifts the untrained model from 0.05 to 0.70, so the experts' gap is mostly which labels
each has seen, not skill.

## 2. Data-parallel merges of the two domain experts

MergeKit linear, TIES, DARE-TIES and task arithmetic (weight 0.5; densities 0.5 / 0.3), for the
original experts and two seed reruns. R is the recovery fraction, (merged - best expert) /
(full - best expert), with paired bootstrap 95% intervals; ranges are over the three seeds.

| method | 4B accuracy | 4B R | 1B accuracy | 1B R |
|---|---|---|---|---|
| **TIES** | **0.897-0.907** | **0.81-0.84** | **0.830-0.838** | **0.70-0.72** |
| DARE-TIES | 0.843-0.867 | 0.68-0.75 | 0.714-0.720 | 0.42-0.45 |
| linear | 0.842-0.861 | 0.68-0.73 | 0.711-0.724 | 0.42-0.46 |
| task arithmetic | 0.842-0.860 | 0.68-0.73 | 0.710-0.724 | 0.42-0.46 |
| full data | 0.977 | 1 | 0.960 | 1 |

Experts' seed SD: 0.003-0.004 (4B), 0.007 (1B). Every merge recovers most of the gap but none
reaches the full model (the best is about 7 points short at 4B). TIES is clearly best here, the
opposite of the summarisation experiments, where it was the weakest merge. Recovery falls at 1B,
most for the methods other than TIES.

## 3. Task-parallel merges

The CLINC150 specialist merged with an EN->DE medical/legal translation specialist (chrF++) and a
constrained-XSum specialist (XSum-C, scored by `self-distill/dialogue_pilot.py`), at weight 1/N.
Retention is the mean over tasks of (score - untrained) / (specialist - untrained); the ceiling is
one model trained on all the tasks' data together.

| 4B | CLINC150 + EN-DE | retention | CLINC150 + XSum-C | retention | all three, retention |
|---|---|---|---|---|---|
| untrained | 0.051 / 18.9 | 0 | 0.051 / 0.588 | 0 | 0 |
| specialists | 0.977 / 59.5 | 1 | 0.977 / 0.790 | 1 | 1 |
| ceiling | 0.973 / 59.6 | 1.00 | 0.971 / 0.802 | 1.03 | 1.02 |
| TIES | 0.962 / 57.3 | 0.96 | 0.971 / 0.443 | 0.14 | -0.03 |
| linear | 0.915 / 57.9 | 0.95 | 0.923 / 0.558 | 0.40 | 0.28 |
| task arithmetic | 0.916 / 58.0 | 0.95 | 0.923 / 0.553 | 0.38 | 0.28 |
| DARE-TIES | 0.920 / 57.6 | 0.95 | 0.928 / 0.578 | 0.45 | 0.24 |

| 1B retention | CLINC150 + EN-DE | CLINC150 + XSum-C | all three |
|---|---|---|---|
| ceiling | 1.00 | 1.00 | 1.00 |
| TIES | 0.88 | 0.69 | 0.58 |
| linear | 0.80 | 0.61 | 0.50 |
| task arithmetic | 0.80 | 0.59 | 0.49 |
| DARE-TIES | 0.79 | 0.62 | 0.48 |

Classification and translation merge well (95-96% retention at 4B). Classification and
summarisation do not: TIES keeps CLINC150 but its XSum-C score (0.443) falls below the untrained
model's (0.588), and the other methods lose some CLINC150 while still falling short on XSum-C.
Retention is lower at 1B throughout, and joint training on all the tasks' data beats every merge.

## 4. Gold-target distillation (4B)

Teacher scores of the gold intents (the teacher sees the label list), student trained with a
cross-entropy weight against the KD term.

| untrained | supervised | KD, CE 1.00 | KD, CE 0.90 | KD, CE 0.75 | KD, CE 0.50 |
|---|---|---|---|---|---|
| 0.051 | 0.977 | 0.977 | 0.975 | 0.975 | 0.962 |

Distillation adds nothing on CLINC150, and a lower CE weight slightly hurts (EN->DE, by contrast:
60.1 chrF++ at CE 0.90 against 59.5 supervised).

## 5. Step-by-step distillation (1B, reduced data)

A 27B teacher, shown each training query and its gold intent, wrote a one-sentence rationale
(4,000 queries); a 1B student was trained on rationale plus intent, against a supervised student on
the intent alone, on the same nested, intent-balanced subsets. Paired bootstrap 95% intervals on
the 4,500 test queries.

| training queries | supervised | step-by-step | difference |
|---|---|---|---|
| 500 | 0.576 | 0.434 | -0.143 [-0.156, -0.130] |
| 1,000 | 0.677 | 0.671 | -0.006 [-0.016, +0.005] |
| 2,000 | 0.854 | 0.864 | +0.009 [+0.000, +0.018] |
| 4,000 | 0.925 | 0.917 | -0.008 [-0.014, -0.001] |

Untrained 1B: 0.052; full data: 0.960. Rationales hurt with 500 queries (the student has to learn
to produce an explanation as well as the label) and make no consistent difference from 1,000 on.

## Conclusions

- CLINC150 is the best case for merging in this project: TIES recovers 81-84% of the domain
  experts' gap at 4B, because the experts differ mainly in which labels they know.
- The best method is task-dependent: TIES for classification, but it sacrifices a generation task
  merged alongside it.
- Recovery and retention fall at 1B.
- Joint multi-task training is the upper bound no merge reaches.
- Neither gold-target nor step-by-step distillation beats supervised training on this task.

## Caveats

- One test set; the domain split is fixed a priori, not sampled.
- The short prompt makes the domain gap one of label coverage; a prompt listing the labels would
  test skill transfer instead.
- Distillation and the step-by-step curves use one training seed per arm.

## Sources (on branch `samsum-scaling`)

- Results: `clinc150-screen.json`, `clinc150-merges.json`, `clinc150-merges-1b.json`,
  `tp-clinc150-*.json`, `tp-1b-clinc150-*.json`, `kd-gold-clinc150-mt_domain.json`,
  `clinc150-sbs-1b.json`.
- Code: `train/prepare_clinc150_data.py`, `train/prepare_multitask_data.py`,
  `merge/clinc150_merge.sbatch`, `merge/lora_merge.sbatch`, `evaluate/clinc150_task.py`,
  `evaluate/merge_report.py`, `evaluate/task_parallel_report.py`, `evaluate/kd_gold_report.py`,
  `self-distill/clinc150_sbs.py`, `evaluate/clinc150_sbs_report.py`.
