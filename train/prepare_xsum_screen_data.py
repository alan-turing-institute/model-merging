# Build XSum data for the base-vs-trained comparison across model sizes, in
# the same shape as the BillSum, SAMSum and CNN/DailyMail screens.
#
# NOT the report's XSum models: those were self-distilled from a teacher. Here
# the trained arm is a supervised LoRA on the reference summaries, like every
# other row of the comparison, so the datasets are read on one protocol.
#
#   full   4,000 training articles (as CNN/DailyMail)
#   test   the programme's standard 1,000: test.shuffle(seed=42), first 1,000
#
# Articles are whitespace-collapsed and cut to 512 words in training AND test -
# the transform behind the report's XSum inputs (prepare_xsum_data.py on
# xsum-summarisation). Prompts come from self-distill/prepare_data_XSum.py, the
# same function xsum_task.py uses, so training and scoring wording match.
#
# Usage, ON A LOGIN NODE:
#   uv run python prepare_xsum_screen_data.py ../datasets/xsum_screen
import argparse
import importlib.util
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

XSUM_HUB = "EdinburghNLP/xsum"
XSUM_REVISION = "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9"
DOCUMENT_WORDS = 512
SEQUENCE_LEN = 1024
TOKENIZER = "google/gemma-3-4b-it"

_spec = importlib.util.spec_from_file_location(
    "prepare_data_XSum", Path(__file__).resolve().parent.parent / "self-distill" / "prepare_data_XSum.py")
_prep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_prep)


def shape(row):
    return {"id": row["id"], "summary": row["summary"].strip(),
            "document": " ".join(row["document"].split()[:DOCUMENT_WORDS])}


def to_training_example(example):
    return {"messages": _prep.convert_to_prompt(example)["messages_short"]
            + [{"role": "assistant", "content": example["summary"]}]}


def main():
    parser = argparse.ArgumentParser(description="Prepare XSum for a base-vs-trained screen.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=1000)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    load = lambda split: load_dataset(XSUM_HUB, split=split, revision=XSUM_REVISION).shuffle(seed=42)

    def collect(split, n):
        kept, dropped = [], 0
        for row in load(split):
            ex = shape(row)
            if not ex["document"] or not ex["summary"]:
                continue
            rendered = tok.apply_chat_template(to_training_example(ex)["messages"], tokenize=False)
            if len(tok(rendered)["input_ids"]) > SEQUENCE_LEN:
                dropped += 1
                continue
            kept.append(ex)
            if len(kept) == n:
                break
        print(f"  {split}: {len(kept)} rows ({dropped} dropped as over {SEQUENCE_LEN} tokens)")
        return Dataset.from_list(kept)

    train, val = collect("train", args.train_size), collect("validation", args.validation_size)
    DatasetDict({"train": train.map(to_training_example),
                 "validation": val.map(to_training_example)}).save_to_disk(str(args.output_dir / "full"))
    (args.output_dir / "ARMS").write_text("full\n")

    test = Dataset.from_list([shape(r) for r in load("test").select(range(args.test_size))])
    test.save_to_disk(str(args.output_dir / "test"))
    ref = sorted(len(tok(s)["input_ids"]) for s in test["summary"])
    print(f"  test: {len(test)} rows; summary tokens p50 {ref[len(ref) // 2]}, max {ref[-1]} "
          f"(xsum_task.py generates up to 128)")


if __name__ == "__main__":
    main()
