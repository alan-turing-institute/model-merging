"""Measure the two KD loss terms directly, on real targets, at correct alignment.

WHY THIS EXISTS. `kd_alpha: 0.2` / `kd_ce_alpha: 1.0` was chosen from measured
magnitudes: cross-entropy converging near 1.4 against a distillation term near
8, so nominal weights of 0.9/0.1 gave the teacher roughly fifty times the
gradient. That reasoning is sound. Its input may not be.

The "near 8" came from axolotl's reported KD loss during training - and every one
of those runs used a teacher misaligned by one position. The same quantity, with
teacher and student the SAME model, sat at 8-20 where it must be zero; corrected,
it is 0.0000. So the figure the weighting was derived from was inflated by the
bug, and how much is unknown.

This measures both terms directly rather than reading them off a training log:
cross-entropy of the student on the gold tokens, and KL(teacher||student) over
the teacher's top-k, renormalised the way the KD kernel does, at the corrected
alignment. No training, one forward pass per model per example.

    uv run --project evaluate python experiments/loss_magnitudes.py \
      --student google/gemma-3-4b-it \
      --teacher google/gemma-3-4b-it --teacher-adapter ../models/gemma3-xsum-full-lora \
      --dataset ../datasets/xsum_dataset/train --limit 32
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from precompute_logprobs import assistant_span  # noqa: E402


def load(model_path, adapter, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", default="google/gemma-3-4b-it")
    ap.add_argument("--student-adapter", default=None)
    ap.add_argument("--teacher", default="google/gemma-3-4b-it")
    ap.add_argument("--teacher-adapter", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--teacher-messages-column", default=None,
                    help="separate context for the teacher (the ctx arm). The "
                         "assistant turn must be identical on both sides - the "
                         "targets are asserted equal and an example is skipped "
                         "otherwise, since the comparison is meaningless if the "
                         "two are scoring different tokens.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    student = load(args.student, args.student_adapter, device, dtype)
    teacher = load(args.teacher, args.teacher_adapter, device, dtype)
    print(f"student {args.student} + {args.student_adapter}")
    print(f"teacher {args.teacher} + {args.teacher_adapter}")
    print(f"device {device} / {dtype}, top-k {args.top_k}, T {args.temperature}\n")

    dataset = load_from_disk(args.dataset).select(range(args.limit))
    ce_all, kl_all, positions = [], [], 0

    skipped = 0
    for row in dataset:
        full_ids, start = assistant_span(tokenizer, row["messages"], args.max_length)
        if full_ids is None:
            continue
        # The teacher may see a different context (the ctx arm). Its logits then
        # come from its own sequence, and are read at its own offset - but only
        # if both sides are predicting the same tokens.
        if args.teacher_messages_column:
            t_ids, t_start = assistant_span(
                tokenizer, row[args.teacher_messages_column], args.max_length)
            if t_ids is None or t_ids[t_start:] != full_ids[start:]:
                skipped += 1
                continue
        else:
            t_ids, t_start = full_ids, start

        with torch.no_grad():
            s_logits = student(input_ids=torch.tensor([full_ids], device=device)).logits[0].float()
            t_logits = teacher(input_ids=torch.tensor([t_ids], device=device)).logits[0].float()

        for offset in range(len(full_ids) - start):
            position = start + offset
            t_position = t_start + offset
            gold = full_ids[position]
            # Position p is predicted by the logits at p-1 - the same frame the
            # producer uses, so this measures what training actually sees.
            s_log = torch.log_softmax(s_logits[position - 1], dim=-1)
            ce_all.append(float(-s_log[gold]))

            t_log = torch.log_softmax(t_logits[t_position - 1] / args.temperature, dim=-1)
            t_top, idx = torch.topk(t_log, k=args.top_k)
            # Renormalise both sides over the teacher's top-k support, which is
            # what the kernel does with a truncated teacher.
            t_top = torch.log_softmax(t_top, dim=-1)
            s_top = torch.log_softmax(s_log[idx], dim=-1)
            kl_all.append(float((t_top.exp() * (t_top - s_top)).sum()))
            positions += 1

    ce = statistics.fmean(ce_all)
    kl = statistics.fmean(kl_all)
    # The kernel scales the KD term by T^2 before weighting.
    kd_term = kl * args.temperature ** 2
    if skipped:
        print(f"skipped {skipped} examples whose teacher/student targets differed\n")
    print(f"{'positions':<26}{positions}")
    print(f"{'cross-entropy (gold)':<26}{ce:.4f}")
    print(f"{'KL(teacher||student)':<26}{kl:.4f}")
    print(f"{'KD term (KL x T^2)':<26}{kd_term:.4f}")
    print(f"{'magnitude ratio KD:CE':<26}{kd_term / ce:.2f} : 1\n")
    print("Gradient contribution at the weightings on record:")
    for a, c in ((0.9, 0.1), (0.5, 0.5), (0.2, 1.0), (0.9, 1.0)):
        share = (a * kd_term) / (a * kd_term + c * ce)
        print(f"  kd_alpha {a} / kd_ce_alpha {c:<4} -> teacher carries "
              f"{100 * share:5.1f}% of the loss  ({a * kd_term:.3f} vs {c * ce:.3f})")


if __name__ == "__main__":
    main()
