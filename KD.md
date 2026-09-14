# Self-distillation onto a merged model

**The question.** The pilot's linear merge of two half-data experts lost to the
better of its own two inputs. A 0.5/0.5 merge yields `base + 0.5·d₁ + 0.5·d₂` —
it halves both task vectors and pulls the model back toward base behaviour.
Does a cheap distillation pass, over data the experts have *already seen*,
recover what merging destroyed?

**The design.** Each half-expert teaches its own half; the student starts from
the merge. `TASK=xsum` (default) or `TASK=crime` selects the arm.

| | XSum (default) | crime |
|---|---|---|
| teacher 1 | `gemma3-xsum-1-of-2-lora` over `xsum_dataset1/train` | `gemma3-crime-1-of-2-lora` over `crime_dataset1/train` |
| teacher 2 | `gemma3-xsum-2-of-2-lora` over `xsum_dataset2/train` | `gemma3-crime-2-of-2-lora` over `crime_dataset2/train` |
| student | `gemma3-xsum-merged-linear` + LoRA | `gemma3-crime-merged-linear-2` + LoRA |
| evaluator | `run_inspect_xsum.py` (ROUGE + length) | `run_inspect_crime.py` (F1 + unparsed) |

**Why XSum is the more informative arm.** Classification can only show the
merge failing one way — predicting the wrong class. On a generation task a
student that has drifted back toward base behaviour shows it *stylistically*
first: output length and sentence count creeping up, preamble returning. Those
move before ROUGE does and are far easier to interpret, and the Inspect
evaluator reports them alongside the score.

The shape matters as much as the result. Nothing here requires any machine to
have seen all the data — the logprob pass shards exactly the way the training
did — so the procedure stays inside the poor-man's-parallelism framing the
project is testing. A distillation that needed the whole dataset in one place
would answer a different and much less interesting question.

## Three regimes

The distillation literature splits by how the teacher is defined. All three are
*self*-distillation in the loose sense that teacher and student share an
architecture; what differs is where the targets come from.

| `KD_MODE` | Teacher | Question it answers |
|---|---|---|
| `offline` (default) | the two half-experts, precomputed, each over its own half | does the experts' knowledge repair the merge? |
| `self` | the merge itself | **control** — or would any distillation pass do? |
| `online` | the student's **own generations**, scored live | does on-policy distillation beat off-policy? |

**The `self` arm is what makes `offline` interpretable.** The crime result —
merge at R = −0.46, distilled at R = +0.78 — is currently open to two readings:
the half-experts restored knowledge the merge had lost, or a second pass over
the data with soft targets at temperature would have helped regardless of who
produced them. Those are very different claims, and only the control separates
them. If `self` recovers as much, the finding is about the distillation
objective and not about merging at all.

### The online arm runs through TRL, not axolotl

Axolotl cannot do this. Its KD trainer requires precomputed teacher logprobs in
every batch — at 0.18.0 *and* on `main` — and the `kd_online_*` config fields
are placeholders the trainer never reads. Its `rl:` key offers
dpo/ipo/kto/simpo/orpo/grpo/ebft and no distillation. So `train_gkd.py` drives
[TRL's `GKDTrainer`](https://huggingface.co/docs/trl/gkd_trainer) instead.

**Why this is a different method, not a delivery detail.** With a frozen
teacher, a merely "live-served" teacher returns *identical* targets to our
precomputed ones — such an arm would compare infrastructure, not methods. GKD
is different in substance: it scores the **student's own generations** rather
than the dataset's gold sequences, which is the train/inference distribution
mismatch the method exists to fix.

| Knob | Meaning |
|---|---|
| `GKD_LMBDA` (1.0) | on-policy fraction; 1.0 = fully student-generated |
| `GKD_BETA` (0.5) | generalized JSD: 0.0 → forward KL, 1.0 → reverse KL |
| `GKD_TEACHER` (`merge`) | `merge` / `full` / `half1` / `half2` |

**On the teacher.** GKD takes one teacher, so the offline arm's
two-teachers-one-half-each design has no direct analogue. The default is the
merge itself — on-policy self-distillation — because it pairs exactly with
`KD_MODE=self` and isolates on-policy from off-policy with the teacher held
constant. `GKD_TEACHER=full` gives an upper bound but **breaks the property
that no machine ever saw all the data**, so it answers a different question and
should be labelled as such.

**Gemma caveat.** TRL warns that Gemma's attention soft-capping yields NaN
logits without a flash-attention implementation. `train_gkd.py` fails on a
non-finite loss rather than training through it — a run that NaNs quietly still
saves an adapter, and the only symptom is a model that generates nothing
coherent, which is exactly how the handed-over Llama model failed.

## Axolotl KD is offline

The trainer does **not** run a teacher. `axolotl.integrations.kd.chat_template`
reads precomputed teacher logprobs out of the dataset, in this schema (taken
from `winglian/evolkit-logprobs-pipeline-75k-v2`, which axolotl's own KD
examples train on):

```
<logprobs_field>: list over ASSISTANT TOKENS of
                  list over top-k of {"logprob": float, "token": "token_id:<int>"}
```

**The dataset `type:` is `kd_strategies.legacy`, a local shim.** axolotl 0.18.0
ships two KD strategies. The module's default `load` returns v2, which expects
a dataset already carrying `target_token_ids` and `target_mask`; `load_legacy`
returns v1, which *builds* those from per-position top-k logprobs — the format
above, and the format of axolotl's own published KD dataset.

No `type:` string reaches `load_legacy`. The resolver in
`axolotl.prompt_strategies.load` does strip a trailing `load_*` and use it as
the function name, but taking that branch skips the branch that corrects the
package — so it tries to import
`axolotl.prompt_strategies.axolotl.integrations.kd.chat_template`, gets
`ModuleNotFoundError`, returns `None`, and the run dies with "unhandled prompt
tokenization strategy". The suffix convention only works for strategies inside
`axolotl.prompt_strategies`.

`train/kd_strategies/legacy.py` exposes `load` and delegates to `load_legacy`.
A dotted path *without* a `load_*` suffix takes the branch that imports the
parent package correctly, so the shim resolves where the direct reference
cannot.

Getting this wrong fails late, inside tokenization, as
`KeyError: 'target_token_ids'` — an error naming neither the strategy nor the
format, which is why it cost three wrong diagnoses before the cause.

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
./run_kd_pipeline.sh                          # XSum
TASK=crime ./run_kd_pipeline.sh
EVAL_LIMIT=100 ./run_kd_pipeline.sh           # shakedown
```

**Prerequisite: real half-experts and a merge for the chosen arm.** The XSum
arm's own pipeline (`run_xsum_pipeline.sh`) has to have produced and registered
them first. This is not a formality — the XSum adapters registered in
`tire-1/tire-2` are 200-example smoke artifacts, which is precisely what the
scale check below exists to catch.

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

### Consuming a merged multimodal model

`gemma-3-4b-it` is multimodal, so axolotl loads an image processor alongside
the tokenizer — whether or not the task involves images. mergekit writes
weights and tokenizer files but not the processor configs, so a merged gemma
model used as a student fails with:

```
OSError: Can't load image processor for '<dir>' ... containing a
preprocessor_config.json file
```

`stage_model_dir.py` stages a directory of **symlinks** to the source and
copies in only the missing processor files from the base model. Symlinks
because the alternative is duplicating ~8GB to add a few kilobytes of JSON, and
because a colleague's handover directory is not writable. The pipeline does
this automatically whenever `MERGE_PATH` points outside `models/`.

## Reading the result

The estimand is the preregistration's recovery fraction, with the distilled
student in place of the merge — on XSum the metric is ROUGE-1, on crime it is
F1, matching each arm's primary metric there:

```
R = (kd − best_half) / (full − best_half)
```

- `R ≥ 1` — distillation recovered the full-data model. The merge was
  repairable, and the pilot's negative result is about merging alone rather
  than about splitting the data.
- `0 < R < 1` — partial repair.
- `R ≤ 0` — no better than keeping the better half and discarding the other,
  which is where the undistilled merge already sits.

The paired test is `experiments/bootstrap_rouge.py` on XSum and
`experiments/mcnemar.py` on crime — ROUGE is a continuous per-example score, so
McNemar does not apply to it.

Evaluation runs through Inspect AI (`evaluate/crime_task.py`), so each run also
leaves a browsable `.eval` log in `$RESULTS_DIR/inspect-logs`:

```bash
inspect view --log-dir "$RESULTS_DIR/inspect-logs"
```

That matters here more than on a metric alone: a distilled student that has
drifted toward one teacher shows it in *which* examples it gets wrong, and the
per-sample view is where that is visible. `CRIME_EVAL=legacy` falls back to
`evaluate.py`; both write the same results JSON.

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
| `train/xsum_gemma_kd.yaml` | KD config, XSum: student = the merge, LoRA on top |
| `train/crime_gemma_kd.yaml` | the same for the crime arm |
| `run_kd_pipeline.sh` | the whole thing |
| `run_kd_after_xsum.sh` | waits for the XSum pipeline to exit, then runs the XSum KD arm with the versions it registered |
| `train_gkd.py` | the online arm — TRL GKDTrainer, on-policy distillation |
