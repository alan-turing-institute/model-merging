"""Split an axolotl-prepared KD dataset into two halves for data-parallel experts.

    python split_prepared.py ../datasets/xsum_kd_screen_gold_prepared ../datasets/xsum_kd_gold_half

kd_lora_train.py takes the largest prepared set as training and the smallest
as validation, so each half is written as <out>_a/train + <out>_a/validation
(and _b). Halves are the first and second 2,000 rows of the prepared order;
both keep the full validation set.
"""
import glob
import sys

from datasets import load_from_disk

src, out = sys.argv[1], sys.argv[2]
sets = [d for d in (load_from_disk(x) for x in glob.glob(f"{src}/*/")) if "target_logprobs" in d.column_names]
train, val = max(sets, key=len), min(sets, key=len)
half = len(train) // 2
for name, rows in (("a", range(half)), ("b", range(half, len(train)))):
    train.select(rows).save_to_disk(f"{out}_{name}/train")
    val.save_to_disk(f"{out}_{name}/validation")
print(f"{src}: {len(train)} train rows -> halves of {half} and {len(train) - half}; {len(val)} validation rows each")
