# Build BillSum training and evaluation data, including the nested subsets that
# make up a data-scaling curve.
#
# WHY BILLSUM. XSum turned out to be saturated for our purposes: every arm -
# base, experts, merges, distilled students - landed inside a ROUGE band narrower
# than the seed-to-seed spread, so no merging question could be answered on it.
# BillSum is harder in the ways that matter here. Its inputs are US Congressional
# bills (median 1,217 words against XSum's ~430) and its references are
# multi-sentence abstractive summaries (median 169 words against XSum's 21), so
# there is far more for a fine-tune to actually learn, and far more room between
# a base model and a trained one for a merge to fall into.
#
# WHAT THE SCALING CURVE IS FOR. Before spending another forty arms on a dataset,
# measure whether this one has headroom at all. Train the same configuration on
# 1/8, 1/4, 1/2 and all of the training data and plot the test score against the
# number of examples. If the curve is still climbing at full data, the task is
# not saturated and differences between merges have somewhere to show up. If it
# has already flattened by half the data, more data - and therefore any method
# that amounts to combining models trained on different data - cannot help, and
# we should learn that now rather than after forty runs.
#
# THE SUBSETS ARE NESTED, deliberately: the 1/8 set is a prefix of the 1/4 set,
# which is a prefix of the 1/2 set, and so on. Four independent draws would
# confound "less data" with "a different draw of data", and with expert-draw
# standard deviations of 0.004-0.005 measured on the crime task, that confound is
# the same size as the effect we are looking for. Nesting removes it: the only
# difference between consecutive arms is the examples the larger one adds.
#
# EVERY ARM SHARES ONE VALIDATION SET, for the same reason - a validation loss
# compared across arms has to be computed on identical rows to mean anything.
#
# Usage:
#   python prepare_billsum_data.py ../datasets/billsum
#   python prepare_billsum_data.py ../datasets/billsum --train-size 4000
import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

# Terse, like STUDENT_SHORT in self-distill/prepare_data_XSum.py: the format is
# meant to be learned into the weights, not re-read from the prompt at every
# call. "Summarise" rather than "Summarize" to match the rest of the repo; the
# references are US federal text but the instruction is ours.
STUDENT_SHORT = "Summarise this US Congressional bill:"
STUDENT_LONG = (
    "Summarise the following US Congressional bill for a legislative digest: state"
    " what the bill does, in the present tense and the third person, covering each"
    " of its substantive provisions in order."
    "\n\nOutput only the summary."
)

# Bills run long: median 1,217 words, 90th percentile 2,071 in the training
# split. Documents are therefore truncated to fit the training window - but in
# TOKENS, not words, and the difference is not cosmetic. Legislative text runs
# 1.74 tokens per word (measured on this split with the Gemma tokenizer), where
# ordinary prose is nearer 1.3: section symbols, subsection markers, dollar
# figures and statutory citations all tokenise badly. A 2400-word cap, which a
# word-count estimate said was ~3,700 tokens, actually produced examples of up to
# 6,948 - and 7% of the full arm was over sequence_len.
#
# THAT IS THE FAILURE THIS FILE EXISTS TO PREVENT. Axolotl's default
# excess_length_strategy is `drop`, so an example longer than sequence_len does
# not become a shortened training example - it becomes no training example at
# all, and the arm quietly contains fewer rows than its name claims.
#
# Verified on axolotl 0.18.0 rather than assumed, because an earlier version of
# this comment claimed the drop was SILENT and that was wrong. Feeding 10 rows
# (7 short, 3 over a 512-token window) through `axolotl preprocess`:
#
#   default              7 of 10 rows kept, and it logs at INFO:
#                        "Dropped 3 sequences outside valid range ([None, 512])"
#   excess_length_strategy: raise      ValueError, job exits nonzero
#   excess_length_strategy: truncate   10 of 10 kept, cut to 512
#
# So the loss IS reported, in one INFO line among a long preprocessing log. The
# config now sets `raise`, which turns that line into a failed job. This script
# is the earlier of the two checks: it fails on the login node during data prep,
# before anything is queued, rather than inside an allocated GPU job.
#
# The budget below is derived from the measured distribution rather than
# estimated: summaries are 214 tokens at the median and 1,101 at the observed
# maximum, and the chat wrapper costs 12. 2900 + 1101 + 12 = 4013, inside a
# 4096 window with room to spare. Summaries are never truncated - the target has
# to survive intact or the example teaches the model to stop early.
DEFAULT_SEQUENCE_LEN = 4096
DEFAULT_DOCUMENT_TOKENS = 2900
TOKENIZER = "google/gemma-3-4b-it"


def truncate_tokens(text: str, tokenizer, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def convert_to_prompt(example):
    """Prompt variants for one bill, keyed as in prepare_data_XSum.py.

    billsum_task.py imports this rather than copying the wording, so the prompt
    a model is evaluated on cannot drift from the one it was trained on.
    """
    document = example["document"]
    return {
        "messages_short": [
            {"role": "user", "content": f"{STUDENT_SHORT}\n\n{document}"},
        ],
        "messages_long": [
            {"role": "user", "content": f"{STUDENT_LONG}\n\nBill:\n\n{document}"},
        ],
    }


def to_training_example(example, prompt_variant: str):
    """The prompt plus the reference summary as the assistant turn."""
    field = f"messages_{prompt_variant}"
    return {"messages": convert_to_prompt(example)[field]
            + [{"role": "assistant", "content": example["summary"]}]}


def normalise(dataset, tokenizer, max_tokens: int):
    """Rename `text` to `document` and truncate it to the token budget.

    BillSum calls the source text `text`; the rest of the repo calls it
    `document`, and the evaluator's entity-grounding scorer reads that name.

    The SAME budget is applied to the training arms and to both evaluation
    splits. It has to be: a model trained on documents cut at 2,900 tokens and
    then scored on whole ones is being asked at test time for something it never
    saw at training time, and the gap would be read as a property of the method
    rather than of the harness.
    """
    renamed = dataset.rename_column("text", "document")
    return renamed.map(
        lambda row: {"document": truncate_tokens(row["document"], tokenizer, max_tokens)},
        desc="truncating documents",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare BillSum, with nested subsets for a data-scaling curve."
    )
    parser.add_argument("output_dir", type=Path, help="Directory to write the datasets into.")
    parser.add_argument(
        "--dataset",
        default="FiscalNote/billsum",
        help="Hub address, or a local directory saved with save_to_disk.",
    )
    # 4000 of the 18,949 available. The curve's job is to show whether the score
    # is still rising, which it can do without exhausting the corpus, and the
    # four arms together are then about two to three GPU-hours rather than ten.
    # Raise it if the curve is still climbing steeply at the top.
    parser.add_argument("--train-size", type=int, default=4000, help="Rows in the full arm.")
    parser.add_argument("--validation-size", type=int, default=500, help="Rows shared by every arm.")
    parser.add_argument("--test-size", type=int, default=1000, help="Rows held out for scoring.")
    parser.add_argument(
        "--prompt-variant", default="short", choices=["short", "long"],
        help="Which prompt the training targets are built with. Evaluate with the same one.",
    )
    parser.add_argument(
        "--document-tokens", type=int, default=DEFAULT_DOCUMENT_TOKENS,
        help="Token budget for the bill text. Must leave room for the longest summary.",
    )
    parser.add_argument(
        "--sequence-len", type=int, default=DEFAULT_SEQUENCE_LEN,
        help="Must match billsum_gemma.yaml. Only used to verify nothing will be dropped.",
    )
    parser.add_argument("--tokenizer", default=TOKENIZER)
    return parser.parse_args()


def verify_fits(dataset, tokenizer, sequence_len: int, label: str) -> int:
    """Count training examples that axolotl would silently drop.

    This is the check the whole file is built around, so it runs on the real
    rendered chat text rather than on an estimate, and it is loud: a single
    over-length example means the arm is not the size its name claims, and the
    scaling curve is measuring something other than data quantity.
    """
    lengths = [
        len(tokenizer(tokenizer.apply_chat_template(row["messages"], tokenize=False))["input_ids"])
        for row in dataset
    ]
    over = sum(length > sequence_len for length in lengths)
    longest = max(lengths)
    print(f"{label:>18}: {len(lengths)} rows, longest {longest} tokens"
          f"{f' - {over} OVER sequence_len={sequence_len}' if over else ' - all fit'}")
    return over


def load(dataset: str) -> DatasetDict:
    if Path(dataset).exists():
        return load_from_disk(dataset)
    return load_dataset(dataset)


def main():
    args = parse_args()
    ds = load(args.dataset)

    # The tokenizer the model will actually train with. Truncating by words
    # instead - the obvious shortcut, since it needs no model - is what put 7%
    # of the full arm over the window on the first attempt.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # BillSum ships train / test / ca_test and no validation split, so the
    # validation rows come off the end of a shuffled train split - after the
    # training pool, so that growing the training arms never eats into them.
    train_pool = normalise(ds["train"], tokenizer, args.document_tokens).shuffle(seed=42)
    needed = args.train_size + args.validation_size
    if needed > len(train_pool):
        raise ValueError(
            f"--train-size {args.train_size} + --validation-size {args.validation_size} "
            f"= {needed} rows, but BillSum train has {len(train_pool)}"
        )
    train_full = train_pool.select(range(args.train_size))
    validation = train_pool.select(range(args.train_size, needed))

    dropped = 0
    scales = {
        "full": args.train_size,
        "half": args.train_size // 2,
        "quarter": args.train_size // 4,
        "eighth": args.train_size // 8,
    }
    for name, n in scales.items():
        # A prefix of one shuffled pool: nested by construction.
        arm = DatasetDict({
            "train": train_full.select(range(n)).map(
                to_training_example, fn_kwargs={"prompt_variant": args.prompt_variant}
            ),
            "validation": validation.map(
                to_training_example, fn_kwargs={"prompt_variant": args.prompt_variant}
            ),
        })
        arm.save_to_disk(str(args.output_dir / name))
        dropped += verify_fits(arm["train"], tokenizer, args.sequence_len, name)

    # Evaluation splits keep `document` and `summary`; billsum_task.py builds the
    # prompt from the document itself, so no `messages` column is needed here.
    #
    # ca_test is California state bills, not federal ones, and its references are
    # roughly twice as long (median 330 words against 152). It is a genuine
    # distribution shift rather than a second sample of the same thing, which
    # makes it the more honest place to look for a merge's advantage: methods that
    # only ever recover in-distribution behaviour have nothing to show on it.
    for split, size in (("test", args.test_size), ("ca_test", None)):
        subset = normalise(ds[split], tokenizer, args.document_tokens).shuffle(seed=42)
        if size is not None:
            subset = subset.select(range(min(size, len(subset))))
        subset.save_to_disk(str(args.output_dir / split))
        print(f"{split:>18}: {len(subset)} rows")

    if dropped:
        raise SystemExit(
            f"\n{dropped} training examples exceed --sequence-len {args.sequence_len} and "
            f"would be SILENTLY DROPPED by axolotl.\nLower --document-tokens (currently "
            f"{args.document_tokens}) or raise sequence_len in billsum_gemma.yaml - and "
            f"change BOTH, they are one decision."
        )


if __name__ == "__main__":
    main()
