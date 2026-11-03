# Off-by-one in Axolotl's KD target alignment

**Affects:** `axolotl==0.18.0`, `axolotl.integrations.kd` (the `chat_template` v2
strategy and the fused Liger KD loss).
**Symptom:** models trained with the KD plugin repeat words immediately —
`a lengthy legal legal battle`, `the opening weekend weekend`, and in the worst
cases `The Lane family family family family family…`.
**Status:** worked around in `generate_teacher_logprobs_XSum.py` by emitting one
extra logprob row per example. Not reported upstream.

## What goes wrong

The KD plugin is never told where the teacher distributions belong. It infers
the offset from lengths alone, in `ChatTemplateStrategyWithKDv2.transform_logprobs`:

```python
input_padding_len = input_seq_len - target_seq_len
```

It then prepends `input_padding_len` masked rows and lays the teacher rows down
in order, so teacher row `j` ends up at index `P + j`, where `P` is the prompt
length. Teacher row `j` is the distribution that produced generated token `j`,
which sits at `input_ids[P + j]`.

So the teacher distribution over token `P + j` is stored at index `P + j`.

But this is a causal language model: the hidden state at index `i` predicts token
`i + 1`. The loss kernel pairs them without any shift —
`target_logprobs_flat[t]` against `student_input_flat[t]` in
`kernels/liger.py` — while correctly shifting the cross-entropy labels right
beside it:

```python
true_labels = torch.nn.functional.pad(true_labels, (0, 1), value=ignore_index)
true_labels_flat = true_labels[:, 1:].contiguous().view(-1)
```

The cross-entropy half is aligned. The KD half is one position late. The model is
trained to make its *next-token* prediction match the distribution of the token
it has **just consumed**, which is a direct recipe for repetition.

This is not a misuse of the API. Axolotl's own reference dataset,
`axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample`, satisfies
`len(logprobs) == len(full) - len(prompt)` exactly, so following it faithfully
reproduces the bug.

## Evidence

Generate a KD dataset where the teacher prompt equals the student prompt and the
teacher **is** the student's initial weights. The teacher distributions are then
the model's own, so the correct alignment must give a KL of zero. Measuring
`KL(teacher || student)` directly against the model, with Axolotl out of the
loop (`check_kd_alignment.py`):

| teacher row `j` placed at | mean KL |
| --- | --- |
| `P + j - 2` | 8.6886 nats |
| **`P + j - 1`** | **0.0204 nats** |
| `P + j` — what Axolotl does | 17.6474 nats |
| `P + j + 1` | 15.3078 nats |
| `P + j + 2` | 14.4421 nats |

Inspecting the prepared dataset Axolotl actually trains on confirms it from the
other direction. Across 12,982 unmasked teacher positions from a real run:

```
teacher top-1 == input_ids[t]      12982  (100.0%)   <- the CURRENT token
teacher top-1 == input_ids[t+1]       37  (  0.3%)   <- the NEXT token
```

And in the generations, against the untrained base model as a control:

| model | outputs with an immediately repeated word | same word 4+ times |
| --- | --- | --- |
| `gemma-3-1b-it` (base) | 0.0% | 0.0% |
| distilled, long prompt | 35.2% | 23.2% |
| distilled, short prompt | 35.0% | 28.6% |

## The fix

Emit **one more logprob row than Axolotl's convention calls for**:

```
len(teacher_logprobs) == len(full_ids) - len(prompt_ids) + 1
```

That makes `input_padding_len` one smaller, so teacher row `j` lands at
`P + j - 1` — the position whose prediction is token `P + j`. The extra row sits
at the final index, whose "next token" does not exist; Axolotl masks it, because
`labels[-1] == -100`, so its contents are irrelevant. A one-hot row keeps the
top-k width uniform, which matters because the plugin derives a single top-k
width per example from the *least frequent* row length.

Verified end to end on the prepared dataset:

```
before fix   top1 == input_ids[t]:   56/56
after fix    top1 == input_ids[t+1]: 56/56
             extra row masked: mask=0, labels=-100
```

## A caveat on the loss scalar

With the fix, a pure-KD run (`kd_ce_alpha: 0.0`, `kd_alpha: 1.0`,
`kd_normalize_topk: true`) on a symmetric dataset reports a loss of ~4.87 rather
than the ~0.02 the direct KL measurement predicts. The gap is *not* alignment —
that is established twice above — and appears to be a roughly constant offset,
since the reported loss moves 12.96 → 4.87 while the measured KL moves
17.65 → 0.02. The most likely explanation is normalisation inside the fused
Liger kernel, so the reported scalar should not be read as a mean per-token KL.
It does not affect gradient direction. This is unresolved.

## How to catch this next time

Run `check_kd_alignment.py` against a **symmetric** dataset (teacher prompt equal
to student prompt, teacher equal to the student's starting weights) before
committing to a long training run. It takes seconds. The equivalent in-training
check is a pure-KD run on the same symmetric data: with correct alignment the
step-0 loss should be near zero, because the model is being asked to match
itself.

Note that this check is only available on a *symmetric* dataset. Once the teacher
is given a different prompt from the student — which is the point of context
distillation — the distributions legitimately differ and no shift reads as zero.
Build the symmetric dataset specifically for the check; do not skip it because
the real dataset is asymmetric.
