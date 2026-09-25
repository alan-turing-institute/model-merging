# This file prepares the XSum dataset for self-distillation training.
# It loads the dataset, samples down to a manageable size, and builds three
# prompt variants per article.
#
# The teacher and the student are deliberately given DIFFERENT prompts. The
# teacher is conditioned on a strict system prompt that reliably produces the
# format we want; the student is trained on a plainer prompt and has to learn
# the format from the weights rather than read it each time. That gap is what
# gives the KD term something to teach -- with an identical prompt on both
# sides, self-distillation sits at a fixed point and teaches nothing.
#
# Three fields are emitted:
#   teacher_messages - what the teacher generates from (strict, system turn)
#   messages_long    - student prompt, the original instruction-heavy version
#   messages_short   - student prompt, terse; the intended end state
#
# The teacher prompt is the "B2_strict" winner of a four-way bake-off: it scored
# 92.5% clean output (no preamble, no markdown, one sentence, 20-25 words)
# against 59.5% for the original prompt. Few-shot variants scored worse and cost
# 2.3x the context, so there are deliberately no exemplars here.
#
# Usage:
#   python prepare_data_XSum.py EdinburghNLP/xsum ../datasets/xsum_prompts
#   python prepare_data_XSum.py path/to/local/xsum out_dir --train-size 20000
import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

# Constraints live in a system turn, as prose. An earlier version used a
# "Rules:" heading with a bullet list and the model echoed the literal label
# into ~6% of its outputs.
TEACHER_SYSTEM = (
    "You are a BBC News sub-editor. You reply with exactly one sentence of 20 to 25 "
    "words, written in the present tense and the third person, in plain text with no "
    "markdown, no asterisks and no quotation marks. You never introduce, label or "
    "explain the sentence, and you never restate the request. Your entire reply is "
    "the sentence itself."
)
# The ask goes AFTER the article so the most recent tokens are the instruction
# rather than several hundred words of news copy.
TEACHER_ASK = "Summarise the article above in one sentence of 20 to 25 words."

STUDENT_LONG = (
    "Summarise the following BBC News article in a single sentence of about 20–25 words,"
    " in the style of a news headline summary: present tense, third person, no preamble."
    "\n\nOutput only the sentence."
)
STUDENT_SHORT = "Summarise this BBC News article in one sentence:"


def convert_to_prompt(example):
    document = example["document"]
    return {
        "teacher_messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"Article:\n\n{document}\n\n{TEACHER_ASK}"},
        ],
        "messages_long": [
            {"role": "user", "content": f"{STUDENT_LONG}\n\nArticle:\n\n{document}"},
        ],
        "messages_short": [
            {"role": "user", "content": f"{STUDENT_SHORT}\n\n{document}"},
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build self-distillation prompts from XSum.")
    parser.add_argument(
        "dataset",
        help="Dataset to load: a local directory saved with save_to_disk, or a Hugging "
        "Face Hub address such as EdinburghNLP/xsum.",
    )
    parser.add_argument("output_dir", type=Path, help="Directory to save the prompts dataset into.")
    # The full dataset has 200k rows, so by default we sample 10k rows for
    # training, 1k for validation, and 1k for testing.
    parser.add_argument("--train-size", type=int, default=10_000, help="Rows to sample for train.")
    parser.add_argument("--validation-size", type=int, default=1_000, help="Rows to sample for validation.")
    parser.add_argument("--test-size", type=int, default=1_000, help="Rows to sample for test.")
    return parser.parse_args()


def load(dataset: str) -> DatasetDict:
    # A path that exists on disk is taken to be a save_to_disk directory;
    # anything else is treated as a Hub address.
    if Path(dataset).exists():
        return load_from_disk(dataset)
    return load_dataset(dataset)


def main():
    args = parse_args()

    # The dataset already has a split into train, validation, and test sets, so we can just use those directly.
    ds = load(args.dataset)

    sample_sizes = {"train": args.train_size, "validation": args.validation_size, "test": args.test_size}
    for split, n in sample_sizes.items():
        if n > len(ds[split]):
            raise ValueError(f"Requested {n} rows from {split!r}, but it only has {len(ds[split])}")

    small_ds = DatasetDict({
        split: ds[split].shuffle(seed=42).select(range(n))
        for split, n in sample_sizes.items()
    })

    # Map the conversion function to the dataset and save the prompts to disk.
    # generate_teacher_logprobs_XSum.py loads this to run teacher generation.
    prompts_ds = small_ds.map(convert_to_prompt)
    prompts_ds.save_to_disk(str(args.output_dir))


if __name__ == "__main__":
    main()
