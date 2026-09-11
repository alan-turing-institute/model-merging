"""Precompute teacher top-k logprobs for axolotl's knowledge-distillation trainer.

Axolotl's KD is OFFLINE: `axolotl.integrations.kd.chat_template` reads teacher
logprobs out of the dataset rather than running a teacher during training. This
script produces them.

THE FORMAT, taken from winglian/evolkit-logprobs-pipeline-75k-v2 (the dataset
axolotl's own KD examples train on) and from the loader source:

    <logprobs_field>: list over ASSISTANT TOKENS of
                      list over top-k of {"logprob": float, "token": "token_id:<int>"}

The loader splits on ":" to recover the id, and truncates every position to the
most common top-k width it sees.

THE THING THAT MATTERS IS ALIGNMENT. The list must have exactly one entry per
assistant token as AXOLOTL tokenizes the example - not as this script happens
to. A silent off-by-one shifts every teacher distribution onto the wrong token
and the run still trains, just on noise. So:

  * the assistant span is derived by rendering the chat twice with the same
    tokenizer and chat template axolotl will use, and taking the difference,
    rather than by any independent tokenisation;
  * the prompt rendering is asserted to be a prefix of the full rendering,
    which is what makes that difference meaningful;
  * --verify re-derives the count independently and reports any mismatch.

Teacher and student share a tokenizer here (both are gemma-3-4b-it, one with an
adapter), so token ids are directly comparable. Distilling ACROSS tokenizers
would need vocabulary mapping and this script would be wrong for it.

Usage (from the repo root, with the evaluate env):
    uv run --project evaluate python precompute_logprobs.py \
      --model google/gemma-3-4b-it \
      --adapter ../models/gemma3-xsum-1-of-2-lora \
      --dataset ../datasets/xsum_dataset1/train \
      --output ../datasets/xsum_kd_half1
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGPROBS_FIELD = "teacher_logprobs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--adapter", default=None, help="Teacher LoRA adapter.")
    parser.add_argument("--dataset", required=True, help="save_to_disk dataset split.")
    parser.add_argument("--output", required=True, help="Where to save_to_disk the result.")
    parser.add_argument("--messages-column", default="messages")
    parser.add_argument("--logprobs-field", default=LOGPROBS_FIELD)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Must match the training config's sequence_len.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify", type=int, default=8,
                        help="Re-check alignment on this many examples (0 to skip).")
    return parser.parse_args()


def load_teacher(model_path, adapter_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device or "auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def encode_chat(tokenizer, messages, add_generation_prompt):
    """Render with the chat template, then encode to a flat list of ids.

    Deliberately not apply_chat_template(tokenize=True): in current
    transformers that returns a BatchEncoding wrapping Encoding objects, so
    len() reports the batch size and indexing yields Encoding instances rather
    than ids - which silently defeats any prefix comparison built on it.
    add_special_tokens=False because the template already emits BOS.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def assistant_span(tokenizer, messages, max_length):
    """Token ids for the whole chat, and where the assistant's tokens start.

    Rendered with the tokenizer's own chat template - the same one axolotl
    applies - so the boundary is defined the same way on both sides.
    """
    prompt_ids = encode_chat(tokenizer, messages[:-1], add_generation_prompt=True)
    full_ids = encode_chat(tokenizer, messages, add_generation_prompt=False)
    # If the prompt is not a prefix of the full render, the difference between
    # them is not the assistant span and everything downstream is meaningless.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None, None
    if len(full_ids) > max_length:
        return None, None
    return full_ids, len(prompt_ids)


@torch.no_grad()
def teacher_logprobs(model, full_ids, start, top_k, device):
    """Top-k logprobs for each assistant token, in order."""
    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids=input_ids).logits[0].float()

    # Position p is predicted by the logits at p-1. The assistant span begins at
    # `start`, so the first distribution of interest is at start-1.
    rows = []
    for position in range(start, len(full_ids)):
        distribution = torch.log_softmax(logits[position - 1], dim=-1)
        values, indices = torch.topk(distribution, k=top_k)
        rows.append(
            [
                {"logprob": float(v), "token": f"token_id:{int(i)}"}
                for v, i in zip(values.tolist(), indices.tolist())
            ]
        )
    return rows


def main():
    args = parse_args()

    dataset = load_from_disk(args.dataset)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_teacher(args.model, args.adapter, device)

    kept, skipped = [], 0
    for row in tqdm(dataset, desc="teacher logprobs", unit="ex"):
        messages = row[args.messages_column]
        full_ids, start = assistant_span(tokenizer, messages, args.max_length)
        if full_ids is None:
            skipped += 1
            continue
        rows = teacher_logprobs(model, full_ids, start, args.top_k, model.device)
        record = {key: row[key] for key in row}
        record[args.logprobs_field] = rows
        record["num_assistant_tokens"] = len(rows)
        kept.append(record)

    if not kept:
        raise SystemExit("No examples survived - check --max-length and the chat template.")

    from datasets import Dataset

    out = Dataset.from_list(kept)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(args.output)

    print(f"\nWrote {len(out)} examples to {args.output}")
    print(f"Skipped {skipped} (prompt not a prefix of the full render, or over --max-length)")
    print(f"Logprobs field: {args.logprobs_field}, top_k={args.top_k}")

    # Two checks, because the obvious one is not sufficient. Counting tokens
    # catches a length mismatch but is blind to an OFF-BY-ONE, which is the
    # failure that matters: every distribution attached to its neighbour still
    # has the right count, still trains, and teaches noise.
    #
    # The second check is the real one. A teacher's top-k for a position should
    # usually contain the token that actually occupies it; shifted by one it
    # should not. Comparing the hit rate at shift 0 against +/-1 separates the
    # two by an order of magnitude, so it detects a shift rather than assuming
    # its absence.
    if args.verify:
        sample = out.select(range(min(args.verify, len(out))))

        bad = 0
        for row in sample:
            messages = row[args.messages_column]
            expected = len(encode_chat(tokenizer, messages, False)) - len(
                encode_chat(tokenizer, messages[:-1], True)
            )
            actual = len(row[args.logprobs_field])
            if expected != actual:
                bad += 1
                print(f"  MISALIGNED: expected {expected} assistant tokens, wrote {actual}")
        print(f"Token-count check: {len(sample) - bad} ok, {bad} mismatched")
        if bad:
            raise SystemExit("Token-count check failed - do not train on this dataset.")

        def hit_rate(shift):
            hits = total = 0
            for row in sample:
                messages = row[args.messages_column]
                full = encode_chat(tokenizer, messages, False)
                start = len(encode_chat(tokenizer, messages[:-1], True))
                for offset, entry in enumerate(row[args.logprobs_field]):
                    position = start + offset + shift
                    if not 0 <= position < len(full):
                        continue
                    ids = {int(e["token"].split(":")[1]) for e in entry}
                    hits += full[position] in ids
                    total += 1
            return hits / total if total else 0.0

        aligned = hit_rate(0)
        shifted = max(hit_rate(-1), hit_rate(1))
        print(f"Alignment check: top-{args.top_k} hit rate {aligned:.3f} at shift 0, "
              f"{shifted:.3f} at +/-1")
        if aligned <= max(3 * shifted, 0.2):
            raise SystemExit(
                "Alignment check FAILED: the teacher's distributions do not track the "
                "tokens they are attached to. Do not train on this dataset."
            )

    widths = {len(position) for row in out.select(range(min(4, len(out))))
              for position in row[args.logprobs_field]}
    print("top-k widths present:", sorted(widths))
    print("Sample entry:", json.dumps(out[0][args.logprobs_field][0][:2]))


if __name__ == "__main__":
    main()
