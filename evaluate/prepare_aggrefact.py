"""The summarisation subsets of LLM-AggreFact (Tang, Laban & Durrett, 2024),
for testing DeepEval's alignment and the LLM judge's consistency as
factual-consistency checkers against the benchmark's human labels.

    cd train && uv run --no-sync python ../evaluate/prepare_aggrefact.py ../datasets/aggrefact

lytang/LLM-AggreFact is gated: accept its terms on Hugging Face (evaluation
only, no training on it) with the account whose token is in
~/.cache/huggingface/token. Writes <out>/<cnn|xsum>_<dev|test>, one row per
(document, claim) with the 0/1 label (1 = supported), from AggreFact-CNN and
AggreFact-XSum.
"""

import sys
from pathlib import Path

from datasets import load_dataset

SUBSETS = {"AggreFact-CNN": "cnn", "AggreFact-XSum": "xsum"}


def main():
    out = Path(sys.argv[1])
    for split in ("dev", "test"):
        ds = load_dataset("lytang/LLM-AggreFact", split=split)
        for name, short in SUBSETS.items():
            rows = ds.filter(lambda r: r["dataset"] == name)
            rows = rows.map(lambda r, i: {"id": f"{short}-{split}-{i}"}, with_indices=True)
            rows.select_columns(["id", "doc", "claim", "label"]).save_to_disk(str(out / f"{short}_{split}"))
            pos = sum(rows["label"])
            words = sorted(len(c.split()) for c in rows["claim"])
            print(f"{name} {split}: {len(rows)} claims, {pos} supported ({pos / len(rows):.0%}); "
                  f"claim words median {words[len(words) // 2]}, max {words[-1]}")


if __name__ == "__main__":
    main()
