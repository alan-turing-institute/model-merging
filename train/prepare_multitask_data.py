"""Concatenate prepared single-task arms into one multi-task training set.

The task-parallel ceiling: one model trained on every task's full data at
once, the multi-task counterpart of a data-parallel full-data model.

    python prepare_multitask_data.py tp_clinc150_mt_domain ../datasets/clinc150/full ../datasets/mt_domain/full

Writes ../datasets/<name>/full/{train,validation} and ARMS. Rows keep only the
`messages` column and are shuffled once (seed 0) so no batch is one task.
"""
import sys
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk

name, sources = sys.argv[1], sys.argv[2:]
parts = [load_from_disk(s) for s in sources]
out = DatasetDict({
    split: concatenate_datasets([p[split].select_columns(["messages"]) for p in parts]).shuffle(seed=0)
    for split in ("train", "validation")
})
root = Path("../datasets") / name
out.save_to_disk(str(root / "full"))
(root / "ARMS").write_text("full\n")
print(f"{name}: " + ", ".join(f"{split} {len(d)}" for split, d in out.items())
      + " from " + ", ".join(f"{s} ({len(p['train'])} train)" for s, p in zip(sources, parts)))
