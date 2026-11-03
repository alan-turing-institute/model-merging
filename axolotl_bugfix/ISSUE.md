# Issue title

KD Loss Becomes Inf/NaN When Student Rules Out A Teacher Top-K Token (normalize_logprobs Underflow)

---

<!-- Paste each section below into the matching field of the GitHub "Bug Report" form. -->

## Expected Behavior

With `kd_trainer: true` and `kd_normalize_topk: true`, the KD loss and its gradients stay finite whatever the student's logits, as long as the teacher's top-k logprobs are finite.

## Current behaviour

If a teacher top-k token is one the student rates more than about 103 nats below its best top-k token, the KD loss is `inf` and the gradients are NaN. One such position in a batch is enough: `grad_norm` becomes `nan`, the optimizer writes NaN into the weights, and from the next step the logged loss is `0` or NaN. Every later eval reports `eval_loss: nan`, and the saved model generates empty strings.

Cause: `normalize_logprobs` in `src/axolotl/integrations/kd/utils.py` renormalizes the top-k by converting to probabilities and back:

```python
position_logprobs_lse = torch.logsumexp(logprobs, dim=-1, keepdim=True)
teacher_probs_t_online = torch.exp(logprobs - position_logprobs_lse)
...
final_logprobs_tensor = torch.log(teacher_probs_t_online)
```

The logsumexp shift protects the sum, but not each entry. In fp32, `exp(x - lse)` is exactly 0 for any entry more than about 103 nats below the row maximum, so that entry becomes `log(0) = -inf`. With `kd_normalize_topk` the student's top-k logprobs go through this function (`kernels/liger.py`, `distillation_loss_fn`). The forward KL term is then `p_teacher * (log p_teacher - (-inf)) = inf`. In the backward pass, `d/dx log(exp(x))` at that entry is `inf * 0 = NaN`.

## Steps to reproduce

Minimal (CPU):

```python
import torch
from axolotl.integrations.kd.utils import normalize_logprobs

print(normalize_logprobs(torch.tensor([[0.0, -1.0, -120.0]]), topk=3))
# tensor([[-0.3133, -1.3133,    -inf]])     expected: [-0.3133, -1.3133, -120.3133]
```

The fused loss gives the same result when one teacher top-k token has a student logit about 200 below the others: `LigerFusedLinearKLTopKLogprobLoss(..., normalize_topk=True)` returns `inf`, and `backward()` gives NaN gradients.

How I hit it: offline KD of `google/gemma-3-4b-it` on 10k XSum summarization rows, with top-20 teacher logprobs from vLLM (`raw_logprobs`). After an apostrophe, the student is almost certain the next token is `s`, but the teacher's top 20 still includes tokens like `▁Trading` at about -25. The student puts those at -108 to -115. With the base model's weights, 2 of 1,000 eval samples and 7 of the first 3,500 training samples hit this. That gives `eval_loss: inf` at step 0, and two separate runs (lr 1e-5 with the default seed, and lr 5e-6 with seed 1234) went NaN at steps 8 and 7. Other runs on similar data survived only by batch order.

## Config yaml

```yaml
base_model: google/gemma-3-4b-it
chat_template: tokenizer_default
plugins:
  - axolotl.integrations.kd.KDPlugin
kd_trainer: true
kd_ce_alpha: 0.5
kd_alpha: 0.5
kd_temperature: 1.0
kd_normalize_topk: true
datasets:
  - path: <local dataset with top-20 teacher logprobs>
    type: axolotl.integrations.kd.chat_template
    field_messages: messages_combined
    logprobs_field: teacher_logprobs
    temperature: 1.0
sequence_len: 4096
sample_packing: false
bf16: true
micro_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 1e-5
lr_scheduler: cosine
warmup_steps: 50
torch_compile: true
```

## Possible solution

Renormalize in log space:

```python
return logprobs - torch.logsumexp(logprobs, dim=-1, keepdim=True)
```

This is mathematically identical (it equals `log_softmax` over the top-k), keeps the -inf padding behaviour, and has a finite gradient (`I - softmax`). I have a PR with this change and regression tests.

Related: `transform_logprobs` in `integrations/kd/chat_template.py` uses the same exp-then-log pattern on the teacher side. It only fails if a teacher logprob is below about -103, which vLLM's full-vocab logprobs rarely are, so I've left it out of the PR.

## Which Operating Systems are you using?

- [x] Linux

## Python Version

3.12.14

## axolotl branch-commit

Hit on 0.18.0 (pip), and still present on main/65f3d3f. torch 2.11.0+cu130, transformers 5.14.1, NVIDIA GH200.
