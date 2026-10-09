# Build SAMSum training and evaluation data, with nested subsets for a
# data-scaling curve.
#
# WHY SAMSum, AND WHAT THE RISK IS. BillSum was chosen because XSum was
# saturated: every XSum arm landed inside a ROUGE band narrower than the
# seed-to-seed spread, so no merging question could be answered on it. SAMSum is
# a different bet from BillSum. BillSum widened the gap by making the TARGET
# long and structured (169-word legislative digests). SAMSum keeps the target
# short - 20.3 words on average, against XSum's 21.0 - and widens the gap by
# moving the INPUT out of distribution instead: messenger dialogue, with a
# summary convention (third person, named participants, reported intent) that an
# instruction-tuned model does not produce unprompted.
#
# Be honest about what that implies. A 20-word target is exactly the profile that
# saturated on XSum, so the plausible failure here is not "no headroom" but
# "headroom exhausted early": a large jump from the base model to the smallest
# arm, and a flat curve above it, because what is being learned is mostly output
# FORMAT, and format is cheap to learn. That is a perfectly good answer - it just
# has to be measured at the bottom of the curve, not the top.
#
# HENCE THE ARMS GO DOWN, NOT UP. BillSum's curve ran 500/1000/2000/4000 and
# found its last real step at 500->1000. Running SAMSum over the same range would
# very likely report four flat steps and tell us nothing about WHERE it
# saturated. The default ladder here is 250/500/1000/2000/4000/8000: six nested
# arms spanning five doublings, which brackets the saturation point instead of
# assuming it is above 500. SAMSum dialogues are short (median 73 words), so six
# arms here cost less GPU time than BillSum's four.
#
# THE SUBSETS ARE NESTED, as in prepare_billsum_data.py and for the same reason:
# the 250 arm is a prefix of the 500 arm, and so on. Independent draws would
# confound "less data" with "a different draw of data", and the expert-draw
# standard deviation measured on the crime task (0.004-0.005) is the same size as
# the effect being looked for.
#
# EVERY ARM SHARES ONE VALIDATION SET. Unlike BillSum, SAMSum ships a real
# validation split (818 rows), so it is taken from there rather than carved off
# the training pool - which also means growing the training arms can never eat
# into it.
#
# Usage, ON A LOGIN NODE (compute nodes have no outbound network):
#   uv run python prepare_samsum_data.py ../datasets/samsum
#   uv run python prepare_samsum_data.py ../datasets/samsum --arms 500,1000,2000
import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

# Terse, matching STUDENT_SHORT in prepare_billsum_data.py and
# self-distill/prepare_data_XSum.py: the format is meant to be learned into the
# weights, not re-read from the prompt at every call.
STUDENT_SHORT = "Summarise this conversation:"
STUDENT_LONG = (
    "Summarise the following conversation in two or three sentences: say who did"
    " what, naming the participants, in the third person and the past tense."
    "\n\nOutput only the summary."
)

# Dialogues are short: 93.8 words at the mean, 73 at the median, 191 at the 90th
# percentile, 803 at the observed maximum. At the ~1.3 tokens per word of
# ordinary prose that is ~1,050 tokens at the very top of the distribution, so a
# 1,200-token document budget truncates nothing in practice and the check below
# confirms it rather than assuming it.
#
# The two numbers are ONE DECISION, as in billsum: axolotl's default
# excess_length_strategy is `drop`, so a budget raised without raising
# sequence_len deletes the long examples instead of shortening them, and the arm
# is then not the size its name claims. samsum_gemma.yaml sets `raise` so that
# fails the job; this script is the earlier check, on the login node.
DEFAULT_SEQUENCE_LEN = 1536
DEFAULT_DOCUMENT_TOKENS = 1200
TOKENIZER = "google/gemma-3-4b-it"

# 250/500/1000/2000/4000/8000. Five doublings, bracketing the saturation point.
DEFAULT_ARMS = [250, 500, 1000, 2000, 4000, 8000]


def arm_name(rows: int) -> str:
    """Arms are named by row count, not by fraction.

    BillSum used full/half/quarter/eighth, which only reads correctly when there
    are exactly four of them and the top of the ladder is "all the data". Here
    the ladder is six long and 8,000 is a little over half of SAMSum's 14,731
    training rows, so a fraction name would be actively misleading.
    """
    return f"n{rows}"


def truncate_tokens(text: str, tokenizer, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def convert_to_prompt(example):
    """Prompt variants for one dialogue, keyed as in prepare_billsum_data.py.

    samsum_task.py imports this rather than copying the wording, so the prompt a
    model is evaluated on cannot drift from the one it was trained on.
    """
    document = example["document"]
    return {
        "messages_short": [
            {"role": "user", "content": f"{STUDENT_SHORT}\n\n{document}"},
        ],
        "messages_long": [
            {"role": "user", "content": f"{STUDENT_LONG}\n\nConversation:\n\n{document}"},
        ],
    }


def to_training_example(example, prompt_variant: str):
    """The prompt plus the reference summary as the assistant turn."""
    field = f"messages_{prompt_variant}"
    return {"messages": convert_to_prompt(example)[field]
            + [{"role": "assistant", "content": example["summary"]}]}


def normalise(dataset, tokenizer, max_tokens: int):
    """Rename `dialogue` to `document` and truncate to the token budget.

    SAMSum calls the source text `dialogue`; the rest of the repo calls it
    `document`, and the evaluator's entity-grounding scorer reads that name.
    Participant names make that scorer unusually informative here - a summary
    that attributes an action to the wrong speaker is the characteristic SAMSum
    failure, and it is an entity error.

    The SAME budget is applied to the training arms and to the test split: a
    model trained on dialogues cut at 1,200 tokens and scored on whole ones is
    being asked at test time for something it never saw at training time.
    """
    renamed = dataset.rename_column("dialogue", "document")
    return renamed.map(
        lambda row: {"document": truncate_tokens(row["document"], tokenizer, max_tokens)},
        desc="truncating dialogues",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare SAMSum, with nested subsets for a data-scaling curve."
    )
    parser.add_argument("output_dir", type=Path, help="Directory to write the datasets into.")
    parser.add_argument(
        "--dataset",
        default="knkarthick/samsum",
        help=(
            "Hub address, or a local directory saved with save_to_disk. The "
            "original Samsung/samsum repo is no longer resolvable; this mirror is."
        ),
    )
    parser.add_argument(
        "--arms", default=",".join(str(n) for n in DEFAULT_ARMS),
        help="Comma-separated training-row counts, smallest first. Nested prefixes.",
    )
    parser.add_argument("--validation-size", type=int, default=500,
                        help="Rows shared by every arm, from SAMSum's own validation split.")
    parser.add_argument("--test-size", type=int, default=500, help="Rows held out for scoring.")
    parser.add_argument(
        "--prompt-variant", default="short", choices=["short", "long"],
        help="Which prompt the training targets are built with. Evaluate with the same one.",
    )
    parser.add_argument("--document-tokens", type=int, default=DEFAULT_DOCUMENT_TOKENS,
                        help="Token budget for the dialogue.")
    parser.add_argument("--sequence-len", type=int, default=DEFAULT_SEQUENCE_LEN,
                        help="Must match samsum_gemma.yaml. Used to verify nothing is dropped.")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    return parser.parse_args()


def verify_fits(dataset, tokenizer, sequence_len: int, label: str) -> int:
    """Count training examples axolotl would drop, on the real rendered chat text."""
    lengths = [
        len(tokenizer(tokenizer.apply_chat_template(row["messages"], tokenize=False))["input_ids"])
        for row in dataset
    ]
    over = sum(length > sequence_len for length in lengths)
    longest = max(lengths)
    print(f"{label:>18}: {len(lengths)} rows, longest {longest} tokens"
          f"{f' - {over} OVER sequence_len={sequence_len}' if over else ' - all fit'}")
    return over


def load(dataset: str):
    if Path(dataset).exists():
        return load_from_disk(dataset)
    # token=False is deliberate. SAMSum is public, and an EXPIRED token in
    # ~/.cache/huggingface/token makes the Hub return 401 on public repos rather
    # than falling back to anonymous access - which reads as "dataset does not
    # exist" and sends you looking for the wrong problem. Gemma is gated and does
    # need the token, but it is already in HF_HUB_CACHE, so this script does not.
    return load_dataset(dataset, token=False)


def main():
    args = parse_args()
    arms = sorted(int(a) for a in args.arms.split(",") if a.strip())
    ds = load(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_pool = normalise(ds["train"], tokenizer, args.document_tokens).shuffle(seed=42)
    if arms[-1] > len(train_pool):
        raise ValueError(
            f"largest arm {arms[-1]} exceeds SAMSum train ({len(train_pool)} rows)"
        )
    validation = (normalise(ds["validation"], tokenizer, args.document_tokens)
                  .shuffle(seed=42).select(range(min(args.validation_size, len(ds["validation"])))))
    validation_rendered = validation.map(
        to_training_example, fn_kwargs={"prompt_variant": args.prompt_variant}
    )

    dropped = 0
    for rows in arms:
        # A prefix of one shuffled pool: nested by construction.
        arm = DatasetDict({
            "train": train_pool.select(range(rows)).map(
                to_training_example, fn_kwargs={"prompt_variant": args.prompt_variant}
            ),
            "validation": validation_rendered,
        })
        name = arm_name(rows)
        arm.save_to_disk(str(args.output_dir / name))
        dropped += verify_fits(arm["train"], tokenizer, args.sequence_len, name)

    # The evaluation split keeps `document` and `summary`; samsum_task.py builds
    # the prompt from the dialogue itself, so no `messages` column is needed.
    test = normalise(ds["test"], tokenizer, args.document_tokens).shuffle(seed=42)
    test = test.select(range(min(args.test_size, len(test))))
    test.save_to_disk(str(args.output_dir / "test"))
    print(f"{'test':>18}: {len(test)} rows")
    print(f"{'validation':>18}: {len(validation)} rows (shared by every arm)")

    if dropped:
        raise SystemExit(
            f"\n{dropped} training examples exceed --sequence-len {args.sequence_len} and "
            f"would be SILENTLY DROPPED by axolotl.\nLower --document-tokens (currently "
            f"{args.document_tokens}) or raise sequence_len in samsum_gemma.yaml - and "
            f"change BOTH, they are one decision."
        )


if __name__ == "__main__":
    main()
