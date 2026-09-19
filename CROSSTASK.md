# Cross-task merging: crime + XSum

The third arm, and the one asking the opposite question to the other two.

[The crime arm](README.md) and [the XSum arm](XSUM.md) both split **one** task
in half, train an expert on each half, and ask whether merging recovers the
full-data model. Both deltas point the same way; the hope is that they add.

This arm merges the two **different** tasks' experts — a Reddit crime
classifier and an XSum summariser — and asks whether one set of weights can
hold both skills, or whether the task vectors interfere.

It is well-posed because the two arms happen to agree on everything that has to
match. Same base `google/gemma-3-4b-it`, same LoRA `r=16`/`alpha=32`, same
`q_proj,k_proj,v_proj,o_proj` targets. Only the training data differs, so each
adapter is cleanly a task vector over one shared base — which is exactly the
object [task arithmetic](https://arxiv.org/abs/2212.04089) assumes.

## What makes the result readable

Every model is evaluated on **both** test sets. That gives four reference
points per task instead of one:

| | |
|---|---|
| **base** | the floor — what the untuned model scores |
| **its own expert** | the ceiling — a model trained only on that task |
| **the other expert** | the cross-task floor: what the merge has to beat, and whether the other task's adapter is actively *harmful* rather than merely irrelevant |
| **the merge** | the experiment |

`experiments/crosstask_table.py` pairs a model's two results back together and
reports **retention** per task:

```
retention = (merged - base) / (expert - base)
```

- `1.0` — keeps everything fine-tuning bought on that task
- `0.0` — back to base behaviour; the merge dropped that skill
- `< 0` — worse than never fine-tuning: active interference

Three outcomes are worth distinguishing, and the raw metrics alone don't:

- **High on both** — the result the method promises.
- **~1.0 on one, ~0.0 on the other** — nothing was merged; the merge picked a
  winner. Easy to mistake for success if you only look at one task.
- **Two middling numbers** — the deltas are fighting.

## Read the format columns first

Interference here is legible in a way it wasn't on the halves experiment,
because the two tasks have incompatible output formats. Crime answers with a
single token (`crime` / `not_crime`); XSum writes a sentence. Leakage shows up
in the format statistics before it shows up in the headline metrics:

- **`unpars`** — crime generations that were neither label. A classifier that
  has started writing sentences.
- **`words`** — mean XSum output length. A summariser collapsing toward
  one-word answers, or drifting back to the base model's 28.9 words.

A merge can post a respectable ROUGE while emitting `not_crime` for a fifth of
the test set. The format columns catch that; ROUGE alone doesn't.

## Running it

```bash
./run_crosstask_pipeline.sh
```

On a GPU compute instance, from the repo root, under `screen`. It fetches both
adapters from the registry, merges them by each method in `MERGE_METHODS`, and
evaluates everything on both tasks.

```bash
MERGE_METHODS="task_arithmetic ties" ./run_crosstask_pipeline.sh
# the full sweep, matching the eight methods the halves arm ran:
MERGE_METHODS="linear task_arithmetic task_arithmetic_w1 ties dare_ties slerp arcee_fusion model_stock" \
  ./run_crosstask_pipeline.sh
CRIME_LORA_VERSION=1 XSUM_LORA_VERSION=3 ./run_crosstask_pipeline.sh
EVAL_LIMIT=100 ./run_crosstask_pipeline.sh                      # quick first pass
MERGE_CACHE_DIR=/tmp/lora-merge-cache-crosstask ./run_crosstask_pipeline.sh
KEEP_MERGES=1 REGISTER_MERGES=0 ./run_crosstask_pipeline.sh
```

**Set the versions deliberately.** Both `gemma3-crime-full-lora` and
`gemma3-xsum-full-lora` carry versions 1–3 from repeated runs of their own
arms, and some of those are smoke-test artifacts registered under production
names. Merging the wrong one silently measures a different experiment. The
defaults are 3 and 3; confirm before trusting a result:

```bash
az ml model list --name gemma3-crime-full-lora --query "[].{v:version,created:creationContext.createdAt}" -o table
```

**Disk.** Same profile as the XSum arm: ~17GB of `--lora-merge-cache` plus
~8.1GB per merge. Each merge's local copy is deleted once both its evaluations
are done, so peak is the cache plus *one* merge, not the cache plus all of
them. Put the cache on `/tmp` if the root disk is tight — `/mnt` itself is
root-owned on an Azure ML compute instance.

**Time it before committing.** Twelve evaluations at the default three methods,
and `evaluate.py` (crime) generates one example at a time with no batching,
unlike `evaluate_summarisation.py`. Run once with `EVAL_LIMIT=100` and scale.

### Reading the results

```bash
uv run --project evaluate python experiments/crosstask_table.py "$RESULTS_DIR"
```

## Caveats

- **Equal weights are not equal influence.** The two adapters were trained on
  different data sizes for different numbers of steps, so their deltas differ
  in magnitude. `0.5/0.5` is a starting point, not a neutral one. The real
  object of interest is the trade-off *curve*, which means sweeping the
  weights — `merge_task_arithmetic_w1_crosstask_config.yaml` is the first step
  (both deltas at full strength) and asymmetric weights are the natural
  follow-up.
- **n=1 per condition, no seed variation.** The crime arm's two half-data
  models differed by 3.7pp from training noise alone. Treat gaps smaller than
  that as unresolved rather than as findings.
- **ROUGE measures conformity to XSum's house style**, not summary quality.
  The base model's zero-shot summaries are often *better* summaries than the
  references; they simply aren't XSum-shaped. A retention number of 0.9 means
  "kept 90% of the house style it learned", not "90% as useful".
- **This arm is self-contained.** It generates its own base and expert
  baselines, so it does not depend on the XSum arm's own evaluations having
  finished.

## Proposed: distillation as an alternative to merging

**Status: design only. Nothing below has been run.**

### The framing changed, and why

This section was first drafted as a repair arm - merge the two experts, then
distil to recover what merging cost. That premise does not survive the evidence.

| setting | linear merge outcome |
|---|---|
| same task, disjoint halves (crime) | **loses to the better half**, R = -0.46 |
| same task, disjoint halves (XSum) | loses slightly, R = -0.34 |
| **different tasks** (crime + XSum) | retains 95-99% of both experts |
| **different domains** (Llama-2 + Meditron, mergekit paper reproduction) | **beats both parents on all six benchmarks** |

Merging two experts trained on *different* things composes well; merging two
experts trained on *the same* thing from disjoint data does not. That is
consistent across two model families and two evaluation stacks, and it has a
plausible mechanism: disjoint-half experts are two conflicting solutions to one
problem and averaging lands between them, where different-task vectors are
closer to orthogonal and add.

So there is little cross-task damage to repair, and a repair arm would be
measuring a gap that barely exists - the same dead end XSum turned out to be.

### The question worth asking instead

**Does distillation compose two skills better than weight averaging does?**

Rather than averaging weights, distil both experts into one student: crime
examples carry crime-expert logprobs, XSum examples carry XSum-expert logprobs,
concatenated into a single training set. `precompute_logprobs.py` already
assigns a teacher per shard, so this needs no new machinery, and unlike the
Llama-2/Meditron pair we hold both experts' actual training data.

This is well-posed in a way the repair framing was not. There is a real baseline
(the linear merge), a real ceiling (each expert's own score on its own task),
and a clear alternative hypothesis (weight averaging is already near-optimal for
orthogonal task vectors, and distillation adds nothing).

### Conditions

Five seeds each. Same base, same two experts, same data, same budget.

| # | Condition | Purpose |
|---|---|---|
| 1 | each expert alone, on both tasks | ceiling on its own task; cross-task floor on the other |
| 2 | linear merge | the method being compared against |
| 3 | **multi-teacher KD from the base** | distillation as an alternative to merging |
| 4 | **multi-teacher KD from the merge** | merge-then-distil; the repair framing, kept because it is one config line |
| 5 | supervised control, `kd_alpha: 0.0`, same data and budget | **the condition that decides whether the teachers matter** |
| 6 | joint training on both tasks' data | what a practitioner would do instead |

Conditions 3 and 4 differ only in `base_model`. Condition 5 is not optional:
on crime it accounted for the large majority of the repair, and the teacher's
measurable contribution turned out to be variance rather than mean. Condition 6
keeps the claim honest - "distillation composes better than merging" is
uninteresting if joint training beats both.

### What must be measured

- **Per task, never pooled.** The cross-task floor here is severe and
  asymmetric: the XSum expert scores 0.45 on crime against the base model's
  0.76, while the crime expert leaves XSum roughly unchanged. A pooled number
  would average a catastrophe against a non-event.
- **Variance across seeds, not only means.** On crime the teacher's clearest
  effect was a 50-fold reduction in seed variance (sd 0.0011 against 0.0078)
  that a table of means would have missed entirely.
- **Degeneracy counts** (`experiments/degeneracy.py`) on the XSum side. A tenth
  of outputs collapsing looks identical to "worse summaries" in ROUGE.
- **Generation length** against the reference distribution.

### Hyperparameters do not carry over

`kd_alpha` and `kd_ce_alpha` encode an assumption about the ratio of two loss
magnitudes. Copying axolotl's 0.9/0.1 cost 0.061 ROUGE-1 on XSum with one
output in ten degenerate, and 0.028 accuracy on crime. The working values there
(0.2/1.0) came from cross-entropy near 1.4 against a top-k KL near 8.

This arm is harder than either: it mixes a single-token classification target
with a ~31-token generative one **in the same batch**, so the two contribute at
different scales within one loss. Measure both terms on the mixture before
choosing weights, and if one pair cannot serve both targets, that is itself a
result worth reporting.

### What would count as a result

Condition 3 or 4 beating condition 2 on **per-task** metrics by more than seed
noise - in mean or in variance - while condition 6 does not beat them at the
same budget. Given that the merge already retains 95-99% of both experts, the
headroom is small: at the provisional n=100 pass the linear merge trailed by
0.05 accuracy on crime and 0.014 ROUGE-1 on XSum. At full sample size those are
roughly eight standard errors and a resolvable bootstrap difference
respectively, so the gap is measurable - but it is a gap of a few points, not
the 24 the eight-task setting produced.

**If the full sweep shows the merge retaining more than it did at n=100, say
so and stop.** A composition method cannot demonstrate value against a baseline
that is already at the ceiling, and that judgement is cheaper to make now than
after five seeds of four conditions.

### Measured: the full sweep (2026-09-18)

Full test sets - 1,077 crime, 1,000 XSum. Retention is against each expert's
own-task score.

| model | crime | XSum R1 | words | retention (crime / XSum) |
|---|---|---|---|---|
| base | 0.7502 | 0.2814 | 31.6 | - |
| crime expert | 0.9768 | 0.2878 | 31.5 | - |
| XSum expert | 0.4596 | 0.4049 | 19.4 | - |
| **task-arithmetic-w1** | **0.9731** | **0.3993** | 18.8 | **99.6% / 98.6%** |
| **ties** | **0.9703** | **0.3994** | 19.1 | **99.3% / 98.6%** |
| arcee-fusion | 0.9694 | 0.3581 | 25.2 | 99.2% / **88.4%** |
| dare-ties | 0.9452 | 0.3951 | 20.6 | 96.8% / 97.6% |
| linear | 0.9424 | 0.3941 | 20.5 | 96.5% / 97.3% |
| task-arithmetic | 0.9415 | 0.3969 | 20.5 | 96.4% / 98.0% |
| slerp | 0.9406 | 0.3937 | 20.5 | 96.3% / 97.2% |
| model-stock | 0.7521 | 0.2812 | 31.5 | 77.0% / 69.4% |

**Two tiers, with a mechanism.** Everything near 96% averages the two task
vectors at weight 0.5, halving each one's magnitude and yielding a diluted
version of both skills. The 99% tier either applies both deltas at full strength
(`task_arithmetic_w1`) or trims conflicts and sign-resolves (`ties`). The task
vectors are near-orthogonal, so adding them whole works where averaging does not.

**`arcee-fusion` is a trap the averages hide.** Near-best on crime at 99.2%, but
only 88.4% of the summariser and 25.2 words against the expert's 19.4 - it keeps
classification by drifting back toward the base model's verbose generation. Only
the length column shows it.

**`model-stock` is a no-op**, returning the base model, as predicted for a
two-model input whose geometric assumption it violates. Drop it from future
two-expert sweeps.

### This stops the composition arm

The stop condition above is met. Against `task_arithmetic_w1` the headroom for a
distilled student is 0.0037 accuracy on crime - under one standard error at
n = 1,077 - and 0.0056 ROUGE-1. **Merging two different-task experts is
free when the method is chosen properly**, and a composition method cannot
demonstrate value against a baseline already at the ceiling.

Comparing a distilled student against *linear* would have shown a 3.4-point
gain. That would have been an artefact of the baseline: linear is joint-worst
here, and the gap it leaves is closed for nothing by a merge method that takes
two minutes and no training. Keeping linear for consistency with the halves arms
was reasonable before this sweep and is not defensible after it.

### The crowding curve: it is displacement per direction, not adapter count

Run 2026-09-18. Holds task count at two and varies adapter count, to separate
weight-space crowding from conflict between tasks.

| n | adapters | method | crime | XSum R1 | words |
|---|---|---|---|---|---|
| 2 | crime-full + xsum-full | task-arith w=1.0 | 0.9731 | 0.3993 | 18.8 |
| **3** | both crime halves + xsum-full | task-arith w=1.0 | **0.5348** | 0.3941 | 18.8 |
| 2 | crime-full + xsum-full | ties | 0.9703 | 0.3994 | 19.1 |
| **3** | both crime halves + xsum-full | ties | **0.9517** | 0.4038 | 19.7 |

reference: base 0.7502 / 0.2814, crime expert 0.9768, XSum expert 0.4049.

**Adding a third adapter destroys `task_arithmetic` at weight 1.0.** Crime falls
0.9731 -> 0.5348, below the untuned base model, and below every component: the
crime half-experts score 0.9294 and 0.9313 individually, and their own linear
merge - the one the halves arm calls a failure at R = -0.46 - still manages
0.9136. TIES, which splits weight across the adapters, loses 0.019.

**The damage is directional, and that is the finding.** Summarisation is
untouched in both merges (0.3941 and 0.4038, generation length unchanged). The
three-adapter set applies *two* crime-pointing adapters at full weight and one
XSum adapter, so the crime direction receives double displacement and collapses
while XSum, at its normal single dose, is unaffected.

So the governing quantity is **total displacement**, which is set by the weights
rather than by the number of models. `task_arithmetic` at weight 1.0 won the
two-way sweep precisely because adding whole vectors preserves both skills - and
that same property makes it fail as soon as a direction is represented twice.

**Normalisation is what controls this, and a separate study confirms it.** A
seven-way merge of heterogeneous Qwen3-4B experts uses weight 0.142857 each -
exactly 1/7, summing to 1 - and `task_arithmetic` holds there (mean 0.7047 over
arc_easy/piqa/hellaswag/mmlu, above six of the seven experts and above the
base). Seven averaged vectors are safe where three accumulated ones are not, so
the rule is about the total, not the count:

| study | weights | total displacement | task_arithmetic |
|---|---|---|---|
| gemma, n = 3 | 1.0 each | ~3x | collapses, 0.9731 -> 0.5348 |
| Qwen3, n = 7 | 1/7 each | ~1x | holds, 0.7047 |

A practitioner merging N adapters should therefore be asking what the weights
sum to, and how many of the vectors point the same way - not what N is.

**What that does not explain is TIES.** At n = 7 it collapses to 0.5147 and
DARE-TIES to 0.4264, on the *same* 1/7 weights that leave linear and
task_arithmetic intact, with `normalize: true` set. So the sparsification
methods fail for a reason unrelated to displacement. One candidate is an outlier
member: the pool includes a Chinese error-correction model scoring 0.4409 on
these English benchmarks, and trim-and-sign-resolve gives every model a vote on
which coordinates survive. Re-merging the seven without it would separate
"TIES does not scale" from "TIES is fragile to a bad member" - one merge and one
evaluation, on assets already registered.

This also qualifies the two-way sweep above: `task_arithmetic_w1` is its best
method on a benchmark where every direction appears once, at weights that
accumulate. That is not a property to rely on. TIES was 0.3 points behind at
n = 2 and survived a third model here - but the Qwen3 result above shows it
failing badly at seven, so "TIES degrades gracefully" does not generalise
either. Neither method is safe by default; the weights and the composition of
the pool decide.

**n = 4 was not run.** Four cached `base+adapter` materialisations need ~34GB of
scratch plus ~8GB for output; the instance's temp disk offers ~29GB once you
account for a 33.6GB swapfile, and the root disk had 23GB free. The 4-way merge
(both directions doubled) would be confirmatory - both tasks should collapse
under `task_arithmetic`, neither under TIES - rather than load-bearing, so it
was skipped rather than worked around. Anyone repeating this should size the
scratch disk at roughly 9GB per adapter before starting.

### What is still open

The **task-diversity** axis. Everything above holds task count at two. Two
different tasks compose for free; the earlier eight-task vision study measured
roughly 24 points of pooled damage. This curve shows that repeated *directions*
crowd, which may be the whole explanation - but it cannot rule out an
independent effect from genuinely distinct tasks, because it never varied them.

Testing that needs new experts: three or four more tasks trained over the same
base with the same LoRA configuration. Training is cheap (~30 minutes each under
QLoRA); the cost is data preparation and an evaluation per task. Keeping them
all classification tasks would let them reuse `crime_task.py` almost unchanged,
which is the difference between a day and a week.

Until that exists, the honest summary is: **merging different experts is free
when each direction appears once, and degrades according to how much any one
direction is over-represented.** No repair method is needed for the settings
measured here, which is why the distillation arm above is stopped.

## Files

| Path | |
|---|---|
| `run_crosstask_pipeline.sh` | fetch both experts, merge, evaluate on both tasks |
| `merge/merge_linear_crosstask_config.yaml` | naive baseline |
| `merge/merge_task_arithmetic_crosstask_config.yaml` | task vectors, 0.5/0.5 |
| `merge/merge_task_arithmetic_w1_crosstask_config.yaml` | both deltas at full strength |
| `merge/merge_ties_crosstask_config.yaml` | trim + sign-resolve — the method built for this case |
| `merge/merge_dare_ties_crosstask_config.yaml` | random drop + rescale, then sign-resolve |
| `merge/merge_slerp_crosstask_config.yaml` | spherical midpoint between the two experts |
| `merge/merge_arcee_fusion_crosstask_config.yaml` | parameter-free importance thresholding |
| `merge/merge_model_stock_crosstask_config.yaml` | parity with the halves sweep; assumption violated here |
| `experiments/crosstask_table.py` | joint table with per-task retention |
