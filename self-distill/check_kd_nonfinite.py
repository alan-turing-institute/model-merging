# Finds the samples whose Axolotl KD loss (or its gradient) is not finite, and
# says which teacher rows cause it.
#
#   uv run python check_kd_nonfinite.py <prepared_dir> [model] [--limit N]
#
# <prepared_dir> is one of Axolotl's tokenised caches, e.g.
# <output_dir>/prepared/<hash>, so the rows are exactly what the trainer saw.
# Each sample is run through the same fused Liger loss the KD trainer uses
# (CE + top-k forward KL, kd_normalize_topk), forward and backward, at the
# model's initial weights.
#
# Suspected mechanism: with normalize_topk the student's top-k logprobs are
# renormalised via log(exp(x - lse)). A top-k entry more than ~100 nats below
# the best one underflows to exp() == 0, so its logprob becomes -inf and the
# KL term p_teacher * (log p_teacher - (-inf)) is inf. The one-hot rows that
# generate_teacher_logprobs.py emits use special tokens 0..20 as fillers,
# which the student may rank that far down.
import argparse
from collections import Counter

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

from axolotl.integrations.kd.kernels.liger import (
    LigerFusedLinearKLTopKLogprobFunction,
    LigerFusedLinearKLTopKLogprobLoss,
)
from axolotl.integrations.kd.utils import normalize_logprobs

# Must match the training config.
KD_CE_ALPHA = 0.5
KD_ALPHA = 0.5
# generate_teacher_logprobs.py's FILLER_LOGPROB; marks a one-hot row.
FILLER_LOGPROB = -30.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_dir", help="An Axolotl prepared-dataset directory.")
    parser.add_argument("model", nargs="?", default="google/gemma-3-4b-it")
    parser.add_argument("--limit", type=int, default=None, help="Only check the first N samples.")
    # The KD trainer builds the loss with Liger's default compiled=True (the
    # per-chunk loss goes through torch.compile), which can change numerics.
    parser.add_argument("--compiled", action="store_true",
                        help="torch.compile the loss, as the trainer does.")
    return parser.parse_args()


def lm_head_of(model):
    base = model.language_model if hasattr(model, "language_model") else model
    return getattr(base, "lm_head", None) or model.lm_head


def main():
    args = parse_args()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    weight = lm_head_of(model).weight.detach()
    kd_loss = LigerFusedLinearKLTopKLogprobLoss(
        KD_CE_ALPHA, KD_ALPHA, 1.0, 0.0, compiled=args.compiled, normalize_topk=True
    )

    ds = load_from_disk(args.prepared_dir)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    bad_samples = []
    bad_rows = Counter()  # (row kind, offset from end of valid rows) -> count
    max_gaps = {"teacher": 0.0, "one-hot": 0.0}
    for i in range(n):
        row = ds[i]
        input_ids = torch.tensor([row["input_ids"]]).cuda()
        ids = torch.tensor([row["target_token_ids"]]).long().cuda()
        lps = torch.tensor([row["target_logprobs"]]).float().cuda()
        mask = torch.tensor([row["target_mask"]]).cuda()
        labels = torch.tensor([row["labels"]]).cuda()

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True,
                        logits_to_keep=1)
        hidden = out.hidden_states[-1].to(weight.dtype).detach().requires_grad_()
        loss = kd_loss(weight, hidden, ids, lps, mask, labels)
        loss.backward()
        loss_ok = bool(torch.isfinite(loss))
        grad_ok = bool(torch.isfinite(hidden.grad).all())

        # Per-row view of the KD term on the valid (unmasked) rows.
        valid = mask[0, :, 0].bool()
        valid_pos = valid.nonzero().flatten()
        with torch.no_grad():
            logits = (hidden[0, valid_pos].float() @ weight.float().T)
            raw = torch.gather(logits, -1, ids[0, valid_pos]) - torch.logsumexp(logits, -1, keepdim=True)
            normed = normalize_logprobs(raw, ids.shape[-1])
            for j, pos in enumerate(valid_pos.tolist()):
                kind = "one-hot" if lps[0, pos, 1] <= FILLER_LOGPROB + 1e-3 else "teacher"
                gap = float(raw[j].max() - raw[j].min())
                max_gaps[kind] = max(max_gaps[kind], gap)
                row_kd = LigerFusedLinearKLTopKLogprobFunction.distillation_loss_fn(
                    logits[j : j + 1], ids[0, pos : pos + 1], lps[0, pos : pos + 1],
                    mask[0, pos : pos + 1],
                )
                if not torch.isfinite(row_kd) or not torch.isfinite(normed[j]).all():
                    bad_rows[(kind, j - len(valid_pos))] += 1

        if not (loss_ok and grad_ok):
            bad_samples.append((i, float(loss), grad_ok))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} checked, {len(bad_samples)} non-finite so far", flush=True)

    print(f"\n{args.prepared_dir} (compiled loss: {args.compiled})")
    print(f"samples checked: {n}")
    print(f"samples with non-finite loss or grad: {len(bad_samples)}")
    for i, value, grad_ok in bad_samples[:20]:
        print(f"  sample {i}: loss {value}, grad finite {grad_ok}")
    print("non-finite rows by (kind, position from end of answer):")
    for (kind, offset), count in sorted(bad_rows.items()):
        print(f"  {kind:8s} {offset:+d}: {count}")
    print("largest gap between best and worst student top-k logprob (nats):")
    for kind, gap in max_gaps.items():
        print(f"  {kind:8s} {gap:.1f}")


if __name__ == "__main__":
    main()
