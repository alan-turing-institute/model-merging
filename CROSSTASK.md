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

## Files

| Path | |
|---|---|
| `run_crosstask_pipeline.sh` | fetch both experts, merge, evaluate on both tasks |
| `merge/merge_linear_crosstask_config.yaml` | naive baseline |
| `merge/merge_task_arithmetic_crosstask_config.yaml` | task vectors, 0.5/0.5 |
| `merge/merge_task_arithmetic_w1_crosstask_config.yaml` | both deltas at full strength |
| `merge/merge_ties_crosstask_config.yaml` | trim + sign-resolve — the method built for this case |
| `experiments/crosstask_table.py` | joint table with per-task retention |
