# This file prepares the Wine Enthusiast reviews dataset for training.
# It loads the dataset, removes duplicate reviews, samples down to a manageable
# size, and converts the data into a format suitable for training a model to
# predict the points rating (80-100) from the review text.
#
# spawn99/wine-reviews looks like the two overlapping Kaggle scrapes merged and
# then split at random: of its 281k rows only 169k descriptions are distinct,
# and over half of the validation and test rows have a description that is also
# in train. So before sampling we drop repeated descriptions within each split,
# then drop validation rows whose description is in train, and test rows whose
# description is in train or validation. That leaves ~137k train, ~12k
# validation and ~20k test rows to sample from.
#
# Usage (from train/):
#   uv run python prepare_data_wine.py
#   uv run python prepare_data_wine.py path/to/local/wine out_dir --train-size 20000
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

PROMPT = (
    "The following is a wine review from Wine Enthusiast magazine. Wines are "
    "scored on a scale from 80 to 100 points. Based on the review, predict "
    "the wine's points rating. Answer with exactly one whole number between "
    "80 and 100, and nothing else."
)


# Convert the datasets to a role/assistant format for training
def convert(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{PROMPT}\n\n{example['description']}"
            },
            {
                "role": "assistant",
                "content": str(example["points"])
            }
        ]
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build points-prediction prompts from wine reviews.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="spawn99/wine-reviews",
        help="Dataset to load: a local directory saved with save_to_disk, or a Hugging "
        "Face Hub address. Defaults to spawn99/wine-reviews.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("../datasets/wine_dataset"),
        help="Directory to save the prompts dataset into. Defaults to ../datasets/wine_dataset.",
    )
    # Even after deduplication there are far more rows than we need, so by
    # default we sample 10k rows for training, 1k for validation, and 1k for testing.
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


def drop_seen(split: Dataset, seen: set) -> Dataset:
    # Keep the first row for each description not already in `seen`, adding
    # each kept description to `seen`. Empty descriptions are dropped too.
    keep = []
    for i, description in enumerate(split["description"]):
        if description and description not in seen:
            seen.add(description)
            keep.append(i)
    return split.select(keep)


def main():
    args = parse_args()

    # The dataset already has a split into train, validation, and test sets, so we can just use those directly.
    ds = load(args.dataset)

    # Deduplicate in order, sharing `seen` across splits so validation excludes
    # anything in train, and test excludes anything in train or validation.
    seen = set()
    ds = DatasetDict({
        split: drop_seen(ds[split], seen)
        for split in ["train", "validation", "test"]
    })

    sample_sizes = {"train": args.train_size, "validation": args.validation_size, "test": args.test_size}
    for split, n in sample_sizes.items():
        if n > len(ds[split]):
            raise ValueError(f"Requested {n} rows from {split!r}, but after deduplication it only has {len(ds[split])}")

    small_ds = DatasetDict({
        split: ds[split].shuffle(seed=42).select(range(n))
        for split, n in sample_sizes.items()
    })

    # Map the conversion function to the dataset and save it to disk
    dataset = small_ds.map(convert)
    dataset.save_to_disk(str(args.output_dir))


if __name__ == "__main__":
    main()
