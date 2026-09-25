# Splits a dataset saved with save_to_disk into n roughly equal, disjoint pieces,
# e.g. so that n students can each be trained on a different slice of the data.
#
# Each piece is saved to <output_dir>/<dataset_name>_<i>_of_<n>. For a
# DatasetDict, the train and validation splits are each split n ways and the
# test split is dropped, so that it stays held out for evaluating the merged
# model rather than being scattered across the pieces.
#
# Usage:
#   python split_dataset.py 4 ../datasets/xsum_prompts ../datasets/xsum_splits
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

DROPPED_SPLITS = {"test"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split a saved dataset into n disjoint pieces.")
    parser.add_argument("n", type=int, help="Number of pieces to split the dataset into.")
    parser.add_argument("dataset_dir", type=Path, help="Directory of a dataset saved with save_to_disk.")
    parser.add_argument("output_dir", type=Path, help="Directory to save the pieces into.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If given, shuffle with this seed before splitting. By default the "
        "existing row order is kept and each piece is a contiguous block.",
    )
    return parser.parse_args()


def shard(ds: Dataset, n: int, i: int, seed: int | None) -> Dataset:
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    # contiguous=True gives pieces whose sizes differ by at most one row.
    return ds.shard(num_shards=n, index=i, contiguous=True)


def main():
    args = parse_args()
    if args.n < 1:
        raise ValueError(f"n must be a positive integer, got {args.n}")

    ds = load_from_disk(str(args.dataset_dir))
    name = args.dataset_dir.resolve().name

    if isinstance(ds, DatasetDict):
        kept = [split for split in ds if split not in DROPPED_SPLITS]
        dropped = [split for split in ds if split in DROPPED_SPLITS]
        if dropped:
            print(f"Dropping split(s): {', '.join(dropped)}")
        if not kept:
            raise ValueError(f"No splits left to divide in {args.dataset_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        if isinstance(ds, DatasetDict):
            piece = DatasetDict({split: shard(ds[split], args.n, i, args.seed) for split in kept})
            sizes = ", ".join(f"{split}={len(piece[split])}" for split in kept)
        else:
            piece = shard(ds, args.n, i, args.seed)
            sizes = f"{len(piece)} rows"

        out_path = args.output_dir / f"{name}_{i + 1}_of_{args.n}"
        piece.save_to_disk(str(out_path))
        print(f"Saved {out_path} ({sizes})")


if __name__ == "__main__":
    main()
