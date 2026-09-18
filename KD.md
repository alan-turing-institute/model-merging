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

**But do not over-trust this check.** It verifies the producer against its own
convention, not against the consumer's. A deliberately shifted dataset was
trained end to end and scored no worse than the aligned one — see the controls
section below. Alignment is necessary, not sufficient, and on XSum it was never
what was wrong.

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

## What the XSum controls established (2026-09-14)

The XSum arm does not work. Three separate students — offline with each half
teaching its own half, self-distillation from the merge, and offline again on
re-derived teacher data — all land at 0.332–0.338 ROUGE-1 against a merge that
scores 0.3970, and all emit empty summaries (18, 23 and 34 of 1000) where the
merge emits none. The degradation is large, reproducible, and significant at
p = 0.0002 on a paired bootstrap over 1000 articles.

Four controls narrow what causes it. Each holds everything fixed but one thing.

**1. The distillation term is the cause, not the pass.** Same student init, same
data, same learning rate, same 500 steps, same trainer and collator, with
`kd_alpha: 0.0` and `kd_ce_alpha: 1.0` — the merge comes through intact:

| model | ROUGE-1 | words | empty |
|---|---|---|---|
| `ce-ctrl` (`kd_alpha: 0.0`) | 0.3983 | 19.2 | 0 |
| merge | 0.3970 | 19.1 | 0 |
| KD student (`kd_alpha: 0.9`) | 0.3375 | 16.8 | 23 |

`ce-ctrl` vs merge: +0.0013, 95% CI [−0.0028, +0.0054], p = 0.53. So the labels,
the chat template, the `eot_tokens` handling, the learning rate and the repair
pass itself are all sound. Turning the teacher signal on is what breaks it.

**2. It is not an off-by-one in the teacher frame.** Axolotl's fused KD kernel
shifts `true_labels` for the cross-entropy term (`kernels/liger.py`) and consumes
the teacher tensors unshifted, which makes the required frame ambiguous. Both
were tried end to end. A student trained on logprobs shifted one position
scores 0.3321 against the unshifted 0.3375 — no improvement, p = 0.072.

**3. It is not top-k truncation.** `transform_logprobs` renormalises the stored
top-k to sum to 1, which would distort a high-entropy generative target if the
top-k covered little mass. Measured over 1651 target positions, the teacher's
top-64 captures a mean of 0.983 (median 0.997); no position falls below 0.5 and
only 0.8% fall below 0.8. The renormalisation is close to a no-op.

**4. The reported KD loss is not a diagnostic.** Cross-entropy alone converges
at 1.36. The KD term sits at 8–20 and does not approach zero even when teacher
and student are the same model — a self-distillation smoke test with the base
model on both sides starts at ~20 and ends at ~18. Do not read the KD loss as a
per-token KL, and do not use it to compare configurations; it cannot separate a
working setup from a broken one. Use held-out ROUGE.

**5. The cause is gradient magnitude.** Dropping the learning rate from 1e-4 to
1e-5, changing nothing else, recovers half the damage:

| model | ROUGE-1 | words | empty |
|---|---|---|---|
| `ce-ctrl` (`kd_alpha: 0.0`) | 0.3983 | 19.2 | 0 |
| merge | 0.3970 | 19.1 | 0 |
| KD @ lr 1e-5 | 0.3641 | 17.9 | 2 |
| KD @ lr 1e-4 | 0.3321 | 17.0 | 34 |

+0.0320 over the 1e-4 student, p = 0.0002, with empty summaries falling from 34
to 2 and mean length moving back toward the merge. A monotone dose-response in
step size is what a too-large effective learning rate looks like, and it also
explains the empties: they were over-stepping, not a malformed target.

The arithmetic is the point. The KD term converges around 8 where cross-entropy
converges at 1.36, and it carries weight 0.9 against CE's 0.1 — so it
contributes on the order of fifty times the gradient, at a learning rate chosen
for a CE-scale loss. **`learning_rate` and `kd_alpha` cannot be set
independently of the loss scale, and the scale depends on target length.** A
value that is a gentle repair on single-token classification targets is a
demolition on ~31-token summarisation ones.

Still unresolved: lr 1e-5 remains 0.0330 below the merge (p = 0.0002), so a
tenfold reduction is not enough. The next points on the curve are lr 1e-6 and
`kd_alpha` 0.3, and neither has been run.

## The crime controls (2026-09-18): the repair is not distillation

Re-run against the ORIGINAL handover artifacts, so every baseline matches the
arm on record to four decimals. Only the loss and the config differ.

| arm | accuracy | R | vs full-data |
|---|---|---|---|
| full data | 0.9694 | 1.00 | — |
| **CE-only control (`kd_alpha: 0.0`)** | **0.9573** | **+0.68** | p = 0.072, not significant |
| offline KD, no `eot_tokens`, lr 1e-4 | 0.9424 | +0.29 | p = 0.00026 |
| offline KD, `eot_tokens`, lr 1e-5 | 0.9294 | -0.05 | — |
| offline KD, `eot_tokens`, lr 1e-4 (current code) | 0.9118 | -0.51 | — |
| better half | 0.9313 | — | — |
| merge | 0.9136 | -0.46 | — |

**Removing the teacher makes the student better.** 0.9573 without distillation
against 0.9424 with it. The control beats the merge (p = 1.3e-08), beats the
better half (p = 6.2e-05), and is indistinguishable from full-data training
(p = 0.072). The distillation term does not merely fail to contribute at 0.9
weight - it costs about 0.015 accuracy.

So the repair is a cheap supervised pass over the union of both halves, and the
teacher signal is not carrying it. The 0.9610 on record sits near this control's
0.9573, which is consistent with the original arm's gain having been its
0.1-weighted cross-entropy term all along.

### Two things this does not show

The control is **not a single-variable ablation**: it sets `kd_ce_alpha` to 1.0
where the KD arm has 0.1, so it removes the teacher and strengthens the label
term tenfold at once. It answers "would a plain supervised pass do better?" -
yes - and not "does the teacher contribute exactly nothing at fixed CE weight?".
The stricter ablation (`kd_alpha: 0.0`, `kd_ce_alpha: 0.1`) has not been run.

And it **weakens the parallelism framing rather than supporting it**. The repair
pass trains on the union. A machine that can run it has all the data already, so
the property that no machine ever sees everything is gone. What survives is a
warm-start argument - sharded experts plus one epoch over the union reaching
parity with three epochs of full training - which may be a real compute saving
but is a different claim from the one the project set out to test.

### fc84af5 broke the crime arm for a week

That commit added `eot_tokens` and `train_on_eos: turn` to the crime KD config,
carrying across the XSum turn-terminator fix. On a single-token classification
target it takes supervision from 1 position to 4, three of which teach the model
to stop rather than to answer, and quadruples the distillation gradient at an
unchanged learning rate. Cost: 0.031 accuracy and 29 unparsable outputs. The fix
was right for XSum and wrong to propagate; the crime arm was never re-run
afterwards, so nobody noticed until the reproduction attempt.

Lowering the learning rate to 1e-5 was a partial remedy (0.9294, no unparsable
outputs). Removing the terminator was the actual fix (0.9424).

### Baselines are not interchangeable across generations

The arm on record evaluated against models in a colleague's share directory; the
registry's v3 generation scores differently on the same test set - half 2 at
0.9619 against 0.9313, a three-point gap larger than the full-vs-half margin
that R normalises by. Recovery fractions computed across the two generations are
not comparable. Always record provenance, and always compare within one
generation.

### Five seeds: the control's spread contains the entire reported effect

Run 2026-09-18, five seeds of the `kd_alpha: 0.0` control, identical otherwise.

| seed | accuracy | R |
|---|---|---|
| 4 | 0.9721 | +1.07 |
| 1 | 0.9610 | +0.78 |
| 3 | 0.9591 | +0.73 |
| 42 | 0.9573 | +0.68 |
| 2 | 0.9508 | +0.51 |
| **mean ± sd** | **0.9601 ± 0.0078** | **+0.76 ± 0.20** |

The control's mean recovery is **+0.76**. The figure on record for knowledge
distillation is **+0.78**. They are the same number, and the control has no
teacher. Seed 1 reproduces 0.9610 to four decimals - the exact accuracy the
distillation arm reported - and seed 4 at 0.9721 beats full-data training
(0.9694) outright.

R ranges +0.51 to +1.07 across seeds of one configuration. That spread of 0.56
is larger than every effect this project has reported except the merge's own
deficit. The crime decomposition compared +0.78 against +0.39 and attributed the
difference to the teachers; two standard deviations here is 0.40.

So: **no comparison at n=1 in this project could support the claims made from
it.** The preregistration asked for five seeds. Anything below about 0.01
accuracy - a quarter of the full-vs-better-half gap that R normalises by - is
indistinguishable from seed noise.

What survives: the merge's deficit, R = -0.46 against a noise scale of ±0.20.
Merging two disjoint-half experts loses to keeping the better half, and a plain
supervised pass over the union repairs it to roughly full-data parity. The
distillation arm at 0.9424 sits 2.3 sd below the control mean and below its
observed minimum, so distillation is, if anything, mildly harmful - though that
arm is itself a single seed and needs its own replication before the magnitude
is quoted.

### Everything else here is a single run

Only the control above has been replicated. Every other number - the KD arms,
the online arm, both XSum arms - is one draw from a distribution whose width we
now know to be about ±0.008 accuracy on this task.

### What this means for the crime arm

The crime arm improved (merge 0.9136 → 0.9610 offline, 0.9461 self, against
0.9694 full-data), and it ran with the same KD term that damages XSum. The
difference between the tasks is target length: crime's targets are a single
label token, XSum's are ~31, so crime receives roughly thirty times less KD
gradient per example. Control 5 confirms that axis directly: the damage
scales with the learning rate, and crime receives roughly thirty times less KD
gradient per example than XSum does.

Until that is tested, **the crime decomposition claim is not safe**. "Half the
repair is the distillation pass and half is the experts' knowledge" assumes the
KD term was contributing what it appeared to. The crime arm needs its own
`kd_alpha: 0.0` control before that sentence goes anywhere.

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
