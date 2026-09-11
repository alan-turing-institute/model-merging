# Self-distillation onto a merged model

**The question.** The pilot's linear merge of two half-data experts lost to the
better of its own two inputs. A 0.5/0.5 merge yields `base + 0.5·d₁ + 0.5·d₂` —
it halves both task vectors and pulls the model back toward base behaviour.
Does a cheap distillation pass, over data the experts have *already seen*,
recover what merging destroyed?

**The design.** Each half-expert teaches its own half; the student starts from
the merge.

| | |
|---|---|
| teacher 1 | `gemma3-crime-1-of-2-lora` over `crime_dataset1/train` |
| teacher 2 | `gemma3-crime-2-of-2-lora` over `crime_dataset2/train` |
| student | `gemma3-crime-merged-linear-2` + LoRA |

The shape matters as much as the result. Nothing here requires any machine to
have seen all the data — the logprob pass shards exactly the way the training
did — so the procedure stays inside the poor-man's-parallelism framing the
project is testing. A distillation that needed the whole dataset in one place
would answer a different and much less interesting question.

## Axolotl KD is offline

The trainer does **not** run a teacher. `axolotl.integrations.kd.chat_template`
reads precomputed teacher logprobs out of the dataset, in this schema (taken
from `winglian/evolkit-logprobs-pipeline-75k-v2`, which axolotl's own KD
examples train on):

```
<logprobs_field>: list over ASSISTANT TOKENS of
                  list over top-k of {"logprob": float, "token": "token_id:<int>"}
```

`precompute_logprobs.py` produces it. The expensive step is therefore a forward
pass of each teacher over its half, not the training.

### Alignment is the thing that can silently ruin this

The logprob list must have exactly one entry per assistant token *as axolotl
re-tokenizes the example*. An off-by-one attaches every teacher distribution to
its neighbour: the run still trains, the loss still falls, and the student
learns noise. There is an [open bug class](https://github.com/axolotl-ai-cloud/axolotl/issues/2978)
in exactly this area.

So `precompute_logprobs.py` verifies rather than assumes. Counting tokens is
not enough — a shifted list has the right length. The real check is that a
teacher's top-k for a position should contain the token that actually occupies
it, and should not when shifted:

```
Alignment check: top-64 hit rate 0.785 at shift 0, 0.094 at +/-1
```

An order of magnitude apart. If shift 0 is not clearly the best, the script
exits rather than writing a dataset you would train on.

## Running it

```bash
./run_kd_pipeline.sh
EVAL_LIMIT=100 ./run_kd_pipeline.sh          # shakedown
```

On a GPU instance under `screen`. Stages: fetch artifacts → **verify teacher
scale** → teacher logprobs per half → combine → distil → register → evaluate
against the merge, both halves and the full-data model.

### The scale check is not optional

`check_adapter_scale.py` runs before anything expensive. It derives the step
count the dataset implies and compares it to what the adapter's checkpoints
record:

```
Expected : ~500 steps (8000 / 16 x 1 epochs)
Recorded : 25  checkpoints=[13, 25]
FAIL: 25 steps is under 50% of the ~500 this dataset implies.
```

This exists because of a real incident. Registry versions accumulate, smoke
runs get registered under production names, and on the XSum arm the artifacts
were selected by name, then by size, then by registration timestamp — all three
wrong. The checkpoint numbers were what finally settled it. A smoke-scale
teacher produces perfectly well-formed logprobs and teaches nothing, and the
only symptom is a disappointing number hours later.

## Reading the result

The estimand is the preregistration's recovery fraction, with the distilled
student in place of the merge:

```
R = (kd − best_half) / (full − best_half)
```

- `R ≥ 1` — distillation recovered the full-data model. The merge was
  repairable, and the pilot's negative result is about merging alone rather
  than about splitting the data.
- `0 < R < 1` — partial repair.
- `R ≤ 0` — no better than keeping the better half and discarding the other,
  which is where the undistilled merge already sits.

`experiments/mcnemar.py` gives the paired test over the results JSONs.

**Status: exploratory.** `PREREGISTRATION.md` licenses three runs and this is
not one of them — see its Deviations entry dated 2026-09-11. This arm generates
a hypothesis; it does not test one. `kd_alpha` in particular is unswept, and it
is the whole trade-off between matching the teachers and matching the labels.

## Files

| Path | |
|---|---|
| `precompute_logprobs.py` | teacher forward pass → top-k logprobs, with the alignment check |
| `check_adapter_scale.py` | refuses smoke-scale artifacts before expensive steps |
| `concat_datasets.py` | combines the two teachers' halves, shuffled |
| `train/crime_gemma_kd.yaml` | KD config: student = the merge, LoRA on top |
| `run_kd_pipeline.sh` | the whole thing |
