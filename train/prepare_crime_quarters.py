"""Split the crime training set four ways, nested inside the existing halves.

WHY FOUR. The two-way arm cannot answer whether merging beats keeping the better
shard. Full-data training beats the better half by only 0.0167 on this task, and
the merge beats that half by 0.0037 - about one expert-draw standard deviation,
from a single pair. There is nothing there to resolve.

The BillSum scaling curve showed why and what fixes it: the denominator of the
recovery fraction is the gap between full-data training and ONE SHARD, so it
grows as the shards get smaller. On BillSum a two-way split left a denominator of
0.0061 whose interval spanned zero, against 0.0228 at four-way and 0.0503 at
eight-way. Four-way is the smallest split that buys a denominator worth dividing
by, and it keeps each expert on 2,153 examples, enough to train.

NESTED, NOT REDRAWN. Each half is split again with the same seed and the same
stratification that made the halves, so q1 + q2 = half 1 exactly. That keeps this
arm comparable to the two-way results already in the report instead of
introducing a second, unrelated partition of the data.
"""
from datasets import DatasetDict, load_from_disk

for half in (1, 2):
    ds = load_from_disk(f"../../datasets/crime_dataset{half}")
    for split in ("train", "validation"):
        if ds[split].features["label"].__class__.__name__ != "ClassLabel":
            raise SystemExit(f"{split} label is not a ClassLabel; cannot stratify")
    parts = {s: ds[s].train_test_split(test_size=0.5, seed=42,
                                       stratify_by_column="label")
             for s in ("train", "validation")}
    for k, side in ((0, "train"), (1, "test")):
        q = (half - 1) * 2 + k + 1
        out = DatasetDict({"train": parts["train"][side],
                           "validation": parts["validation"][side]})
        out.save_to_disk(f"../../datasets/crime_dataset_q{q}")
        print(f"q{q}: {len(out['train'])} train, {len(out['validation'])} validation")
