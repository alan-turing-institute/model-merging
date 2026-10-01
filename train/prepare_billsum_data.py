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

# Bills run long: median 1,217 words, 90th percentile 2,071, maximum 2,697 in the
# training split. This cap and the configs' `sequence_len: 4096` are chosen
# together and have to move together. Axolotl SILENTLY DROPS any example longer
# than sequence_len, so an uncapped document does not produce a truncated
# training example - it produces no training example at all, and a scaling curve
# whose arms quietly contain fewer rows than their names claim. Truncating here
# instead keeps every row, and keeps the count honest.
#
# 2400 words plus a ~340-word summary is about 3,700 Gemma tokens with the chat
# wrapper, which fits 4096 with room to spare, and leaves ~95% of training
# documents untouched.
MAX_DOCUMENT_WORDS = 2400


def truncate(text: str, max_words: int = MAX_DOCUMENT_WORDS) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


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


def normalise(dataset):
    """BillSum calls the source text `text`; the rest of the repo calls it
    `document`, and the evaluator's entity-grounding scorer reads that name."""
    renamed = dataset.rename_column("text", "document")
    return renamed.map(lambda row: {"document": truncate(row["document"])})


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
    return parser.parse_args()


def load(dataset: str) -> DatasetDict:
    if Path(dataset).exists():
        return load_from_disk(dataset)
    return load_dataset(dataset)


def main():
    args = parse_args()
    ds = load(args.dataset)

    # BillSum ships train / test / ca_test and no validation split, so the
    # validation rows come off the end of a shuffled train split - after the
    # training pool, so that growing the training arms never eats into them.
    train_pool = normalise(ds["train"]).shuffle(seed=42)
    needed = args.train_size + args.validation_size
    if needed > len(train_pool):
        raise ValueError(
            f"--train-size {args.train_size} + --validation-size {args.validation_size} "
            f"= {needed} rows, but BillSum train has {len(train_pool)}"
        )
    train_full = train_pool.select(range(args.train_size))
    validation = train_pool.select(range(args.train_size, needed))

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
        print(f"{name:>8}: {n} train rows")

    # Evaluation splits keep `document` and `summary`; billsum_task.py builds the
    # prompt from the document itself, so no `messages` column is needed here.
    #
    # ca_test is California state bills, not federal ones, and its references are
    # roughly twice as long (median 330 words against 152). It is a genuine
    # distribution shift rather than a second sample of the same thing, which
    # makes it the more honest place to look for a merge's advantage: methods that
    # only ever recover in-distribution behaviour have nothing to show on it.
    for split, size in (("test", args.test_size), ("ca_test", None)):
        subset = normalise(ds[split]).shuffle(seed=42)
        if size is not None:
            subset = subset.select(range(min(size, len(subset))))
        subset.save_to_disk(str(args.output_dir / split))
        print(f"{split:>8}: {len(subset)} rows")


if __name__ == "__main__":
    main()
