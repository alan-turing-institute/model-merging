# Build CNN/DailyMail data for a base-vs-trained comparison across model sizes.
#
# One training arm, `full`: 4,000 articles, the same training-set size as the
# top of the BillSum curve. The whole training split (287k articles) would
# take days of GPU time per model, and the question here is the gap between
# an untrained and a trained model at each size, not the corpus ceiling.
#
# Highlights are kept with their line breaks: each line is one highlight, and
# rougeLsum scores per sentence, so the format matters. Articles are truncated
# to a TOKEN budget with the Gemma tokeniser; any example still over
# sequence_len is dropped and counted rather than silently cut.
#
# abisee/cnn_dailymail 3.0.0, Apache-2.0. Train is streamed (it is large);
# validation and test are small enough to shuffle exactly.
#
# With --ladder it instead writes a data-scaling curve: one arm per size, each
# a PREFIX of the next larger one (nested, as in prepare_samsum_data.py), so
# consecutive arms differ only in the rows the larger adds. The rows come from
# the same shuffled stream as `full`, so the 4,000-row arm is the `full` arm.
# Every arm shares one validation set; the test set is the screen's.
#
# Usage, ON A LOGIN NODE:
#   uv run python prepare_cnndm_data.py ../datasets/cnndm
#   uv run python prepare_cnndm_data.py ../datasets/cnndm_curve \
#       --ladder 8000,4000,2000,1000,500,250,125
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

DATASET, CONFIG = "abisee/cnn_dailymail", "3.0.0"
INSTRUCTION = "Write the highlights for this news article:"
SEQUENCE_LEN = 2048
DOCUMENT_TOKENS = 1600
TOKENIZER = "google/gemma-3-4b-it"


def convert_to_prompt(example):
    """One prompt; cnndm_task.py imports this."""
    return {"messages": [{"role": "user", "content": f"{INSTRUCTION}\n\n{example['document']}"}]}


def to_training_example(example):
    return {"messages": convert_to_prompt(example)["messages"]
            + [{"role": "assistant", "content": example["summary"]}]}


def shape(row, tokenizer):
    ids = tokenizer(row["article"], add_special_tokens=False)["input_ids"]
    return {"id": row["id"], "summary": row["highlights"].strip(),
            "document": tokenizer.decode(ids[:DOCUMENT_TOKENS], skip_special_tokens=True)}


def collect(rows, n, tokenizer, label):
    kept, dropped = [], 0
    for row in rows:
        ex = shape(row, tokenizer)
        rendered = tokenizer.apply_chat_template(to_training_example(ex)["messages"], tokenize=False)
        if len(tokenizer(rendered)["input_ids"]) > SEQUENCE_LEN:
            dropped += 1
            continue
        kept.append(ex)
        if len(kept) == n:
            break
    print(f"  {label}: {len(kept)} rows ({dropped} dropped as over {SEQUENCE_LEN} tokens)")
    return Dataset.from_list(kept)


def main():
    parser = argparse.ArgumentParser(description="Prepare CNN/DailyMail for a base-vs-trained screen.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--ladder", help="comma-separated arm sizes for a nested scaling curve")
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ladder = sorted({int(n) for n in args.ladder.split(",")}, reverse=True) if args.ladder else []
    train_size = ladder[0] if ladder else args.train_size

    train_stream = load_dataset(DATASET, CONFIG, split="train", streaming=True).shuffle(seed=42, buffer_size=10_000)
    train = collect(train_stream, train_size, tok, "train")
    if len(train) < train_size:
        raise SystemExit(f"only {len(train)} training rows collected, wanted {train_size}")
    val = collect(load_dataset(DATASET, CONFIG, split="validation").shuffle(seed=42), args.validation_size, tok, "validation")
    val = val.map(to_training_example)
    # Largest first, so array index 0 is the arm hardest to infer from the others.
    arms = {f"n{n}": n for n in ladder} if ladder else {"full": train_size}
    for name, n in arms.items():
        DatasetDict({"train": train.select(range(n)).map(to_training_example),
                     "validation": val}).save_to_disk(str(args.output_dir / name))
    (args.output_dir / "ARMS").write_text("".join(f"{name}\n" for name in arms))
    print(f"  arms: {', '.join(arms)} (nested prefixes of one shuffled stream)")

    test = load_dataset(DATASET, CONFIG, split="test").shuffle(seed=42).select(range(args.test_size))
    test = Dataset.from_list([shape(r, tok) for r in test])
    test.save_to_disk(str(args.output_dir / "test"))
    ref = sorted(len(tok(s)["input_ids"]) for s in test["summary"])
    print(f"  test: {len(test)} rows; highlights tokens p50 {ref[len(ref) // 2]}, "
          f"p99 {ref[int(.99 * len(ref))]}, max {ref[-1]} - cnndm_task.py max_tokens must clear it")


if __name__ == "__main__":
    main()
