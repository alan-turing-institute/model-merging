"""Teacher==student smoke test for the KD logprob alignment.

If the teacher and the student are the same model, the teacher's distribution at
every supervised position is BY CONSTRUCTION the student's own distribution, so
KL(teacher||student) must be ~0. Anything else is a plumbing fault, and the size
of it says how badly the rows are placed.

This reproduces axolotl's placement rule rather than running axolotl:

    input_padding_len = len(input_ids) - len(teacher_logprobs)
    row j is the target for the student's prediction at index input_padding_len + j
    that prediction is over the token at index input_padding_len + j + 1

and evaluates it under both conventions - one row per assistant token (what this
repo emitted until 6f923a1) and one extra row (the fix) - so the two numbers sit
side by side.

KD.md's control #4 ran this test informally, saw the KD term sit at 8-20 with
identical teacher and student, and concluded the reported loss was not a
per-token KL. It was. This is that control, done properly.

    uv run --project evaluate python experiments/kd_alignment_smoke.py --examples 8
"""
import argparse
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from precompute_logprobs import assistant_span  # noqa: E402  same span logic as the producer


def rows_for(logits, start, end, top_k):
    """Top-k (logprob, id) per position in [start, end), scored from logits[p-1]."""
    out = []
    for position in range(start, end):
        distribution = torch.log_softmax(logits[position - 1], dim=-1)
        values, indices = torch.topk(distribution, k=top_k)
        out.append((values, indices))
    return out


def mean_kl(rows, logits, seq_len):
    """Mean KL(teacher||student) under axolotl's inferred placement.

    Restricted to the teacher's top-k support and renormalised on both sides,
    which is what the KD kernel does with a truncated teacher.
    """
    padding = seq_len - len(rows)
    totals, counted = 0.0, 0
    for j, (values, indices) in enumerate(rows):
        index = padding + j
        if not 0 <= index < seq_len or index + 1 >= seq_len:
            continue  # the final row scores a token that does not exist
        teacher = torch.log_softmax(values, dim=-1)              # renormalised top-k
        student_full = torch.log_softmax(logits[index], dim=-1)
        student = torch.log_softmax(student_full[indices], dim=-1)
        totals += float((teacher.exp() * (teacher - student)).sum())
        counted += 1
    return totals, counted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--dataset", default="../datasets/xsum_dataset/train")
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"model {args.model} on {device} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    dataset = load_from_disk(args.dataset).select(range(args.examples))

    sums = {"fixed": [0.0, 0], "old": [0.0, 0]}
    for row in dataset:
        span = assistant_span(tokenizer, row["messages"], args.max_length)
        full_ids, start = span
        if full_ids is None:
            print("  skipped:", start)
            continue
        L = len(full_ids)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([full_ids], device=device)).logits[0].float()

        for name, end in (("fixed", L + 1), ("old", L)):
            total, n = mean_kl(rows_for(logits, start, end, args.top_k), logits, L)
            sums[name][0] += total
            sums[name][1] += n
        print(f"  example: {L} tokens, {L - start} assistant")

    print()
    print(f"{'convention':<34}{'rows':>8}{'positions':>11}{'mean KL (nats)':>17}")
    print("-" * 70)
    for name, label in (("fixed", "one extra row (6f923a1)"),
                        ("old", "one row per assistant token")):
        total, n = sums[name]
        print(f"{label:<34}{'L-P+1' if name == 'fixed' else 'L-P':>8}{n:>11}"
              f"{total / n if n else float('nan'):>17.4f}")
    print()
    print("Teacher and student are the same model, so a correct alignment must give ~0.")


if __name__ == "__main__":
    main()
