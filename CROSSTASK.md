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

### Provisional numbers this design rests on

From the `EVAL_LIMIT=100` timing pass, 2026-09-18. **Not yet confirmed at full
sample size** - the full sweep supersedes these.

| model | crime acc | XSum ROUGE-1 |
|---|---|---|
| base | 0.7600 | 0.2701 |
| crime expert | 0.9700 | 0.2817 |
| XSum expert | 0.4500 | 0.4088 |
| linear merge | 0.9200 | 0.3948 |

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
