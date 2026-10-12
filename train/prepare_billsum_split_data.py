# BillSum two-expert screen at the steep end of its learning curve.
#
# BillSum has no natural partition in its training data, so the merging
# denominator has to come from operating where the curve still rises. The
# 2026 scaling curve put the 500 -> 1,000 step at +0.0275 rougeLsum and the
# 2,000 -> 4,000 step at +0.0061 (flat); this screen measures the denominator
# directly at full = 1,000 with two real experts on COMPLEMENTARY random halves
# of 500 - the experiment a two-way merge would actually run.
#
# Arms: full (1,000), expert_1 and expert_2 (disjoint 500s). No iid_half arm:
# with random experts it would duplicate them. Prompt, truncation and the
# fits-the-window check are imported from prepare_billsum_data.py so the screen
# cannot drift from the curve it follows up.
#
# Usage, ON A LOGIN NODE:
#   uv run python prepare_billsum_split_data.py ../datasets/billsum_split
import argparse
import importlib.util
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

spec = importlib.util.spec_from_file_location("prep", Path(__file__).with_name("prepare_billsum_data.py"))
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--full-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(prep.TOKENIZER)
    ds = load_dataset("FiscalNote/billsum")

    pool = prep.normalise(ds["train"], tok, prep.DEFAULT_DOCUMENT_TOKENS).shuffle(seed=42)
    full = pool.select(range(args.full_size))
    val = pool.select(range(args.full_size, args.full_size + args.validation_size))
    render = lambda d: d.map(prep.to_training_example, fn_kwargs={"prompt_variant": "short"})
    half = args.full_size // 2
    arms = {"full": full, "expert_1": full.select(range(half)),
            "expert_2": full.select(range(half, args.full_size))}
    over = 0
    for name, rows in arms.items():
        arm = DatasetDict({"train": render(rows), "validation": render(val)})
        arm.save_to_disk(str(args.output_dir / name))
        over += prep.verify_fits(arm["train"], tok, prep.DEFAULT_SEQUENCE_LEN, name)
    (args.output_dir / "ARMS").write_text("\n".join(arms) + "\n")
    test = prep.normalise(ds["test"], tok, prep.DEFAULT_DOCUMENT_TOKENS).shuffle(seed=42)
    test.select(range(args.test_size)).save_to_disk(str(args.output_dir / "test"))
    print(f"{'test':>18}: {args.test_size} rows")
    if over:
        raise SystemExit(f"{over} examples over sequence_len - see prepare_billsum_data.py")


if __name__ == "__main__":
    main()
