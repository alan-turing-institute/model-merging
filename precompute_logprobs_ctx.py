"""Teacher logprobs for the CONTEXT-DISTILLATION arm (XSUM_VARIANT=ctx).

`precompute_logprobs.py` assumes one message list: the teacher and the student
see the same prompt, and the teacher's edge is its weights. Here the teacher is
conditioned on a strict system prompt and the student sees a terse one, so the
two sequences differ and the rows have to be transplanted from one to the other.

WHAT MAKES THAT LEGITIMATE. The assistant turn is byte-identical in both
renderings, so its token ids are identical too - only the preceding context
differs. A distribution the teacher produced over target token k is therefore a
distribution over the same token the student must predict at its own position k.
This script asserts that equality per example and skips anything that fails it,
because if the target tokenisations ever diverge the transplant is silently
meaningless.

THE ROW COUNT IS THE STUDENT'S, NOT THE TEACHER'S. Axolotl infers placement from
the count (see teacher_logprobs in precompute_logprobs.py): it needs
`len(student_ids) - student_start + 1` rows so the base offset lands at
student_start - 1. The teacher span has the same length by the assertion above,
so emitting one row per teacher target position plus the trailing filler gives
exactly that - but it is checked rather than assumed.

    uv run --project evaluate python precompute_logprobs_ctx.py \
      --model google/gemma-3-4b-it \
      --adapter ../models/gemma3-xsum-1-of-2-lora \
      --dataset ../datasets/xsum_ctx_dataset1/train \
      --output ../datasets/xsum_ctxkd_half1 --verify 32
"""
import argparse
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from precompute_logprobs import (
    ALIGNMENT_MARKER,
    LOGPROBS_FIELD,
    encode_chat,
    load_teacher,
    teacher_logprobs,
)


def spans(tokenizer, teacher_messages, student_messages, max_length):
    """(teacher_ids, teacher_start, student_len, student_start) or (None, reason)."""
    t_prompt = encode_chat(tokenizer, teacher_messages[:-1], add_generation_prompt=True)
    t_full = encode_chat(tokenizer, teacher_messages, add_generation_prompt=False)
    s_prompt = encode_chat(tokenizer, student_messages[:-1], add_generation_prompt=True)
    s_full = encode_chat(tokenizer, student_messages, add_generation_prompt=False)

    if t_full[: len(t_prompt)] != t_prompt or s_full[: len(s_prompt)] != s_prompt:
        return None, "prompt-not-a-prefix"
    if len(t_full) > max_length or len(s_full) > max_length:
        return None, "over-max-length"
    if len(t_full) <= len(t_prompt) or len(s_full) <= len(s_prompt):
        return None, "no-target-tokens"
    # The transplant is only meaningful if both sides are predicting the same
    # tokens. Same text, same template suffix - but check, do not assume: a
    # tokenizer that merges across the prompt boundary would break it silently.
    if t_full[len(t_prompt):] != s_full[len(s_prompt):]:
        return None, "target-tokens-differ"
    return (t_full, len(t_prompt), len(s_full), len(s_prompt)), None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--verify", type=int, default=0)
    p.add_argument("--logprobs-field", default=LOGPROBS_FIELD)
    p.add_argument("--messages-column", default="messages")
    p.add_argument("--teacher-column", default="teacher_messages")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = load_from_disk(args.dataset)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.teacher_column not in dataset.column_names:
        raise SystemExit(
            f"{args.dataset} has no '{args.teacher_column}' column - build it with "
            f"XSUM_VARIANT=ctx python prepare_xsum_data.py"
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_teacher(args.model, args.adapter, device)

    kept, skipped = [], Counter()
    for row in tqdm(dataset, desc="teacher logprobs (ctx)", unit="ex"):
        span, reason = spans(tokenizer, row[args.teacher_column],
                             row[args.messages_column], args.max_length)
        if span is None:
            skipped[reason] += 1
            continue
        t_full, t_start, s_len, s_start = span
        rows = teacher_logprobs(model, t_full, t_start, args.top_k, model.device)

        # The count axolotl needs is defined by the STUDENT sequence. Equal by
        # construction given the target-token assertion, so a mismatch here means
        # that assertion has been weakened - fail loudly rather than write it.
        expected = s_len - s_start + 1
        if len(rows) != expected:
            raise SystemExit(
                f"row count {len(rows)} != student span + 1 ({expected}); the "
                f"teacher/student target alignment is broken"
            )

        record = {key: row[key] for key in row}
        record[args.logprobs_field] = rows
        record["num_assistant_tokens"] = len(rows) - 1
        kept.append(record)

    if not kept:
        raise SystemExit("No examples survived - check --max-length and the prompts.")

    out = Dataset.from_list(kept)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(args.output)
    (Path(args.output) / ALIGNMENT_MARKER).write_text(
        "context-distillation: teacher rows scored under teacher_messages, "
        "placed against messages; assistant targets asserted identical\n"
    )

    print(f"\nWrote {len(out)} examples to {args.output}")
    if skipped:
        detail = ", ".join(f"{c} {r}" for r, c in sorted(skipped.items()))
        print(f"Skipped {sum(skipped.values())}: {detail}")
    print(f"Logprobs field: {args.logprobs_field}, top_k={args.top_k}")

    if args.verify:
        sample = out.select(range(min(args.verify, len(out))))
        bad = 0
        for row in sample:
            s_full = encode_chat(tokenizer, row[args.messages_column], False)
            s_prompt = encode_chat(tokenizer, row[args.messages_column][:-1], True)
            if len(row[args.logprobs_field]) != len(s_full) - len(s_prompt) + 1:
                bad += 1
        print(f"Student-span check: {len(sample) - bad} ok, {bad} mismatched")
        if bad:
            raise SystemExit("Student-span check failed - do not train on this dataset.")


if __name__ == "__main__":
    main()
