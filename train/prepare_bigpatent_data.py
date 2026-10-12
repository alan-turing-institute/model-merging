# Build BigPatent data for a domain-split merging headroom screen.
#
# WHY BIGPATENT. Of the summarisation candidates checked, it is the only one with
# a natural non-IID partition that also loads cleanly (CC BY 4.0, parquet, no
# script): one config per CPC section. XSum and SAMSum saturated within a few
# hundred rows and have no partition; BillSum has no partition at all. Targets
# are multi-sentence abstracts (~150 words), not one-line headlines.
#
# SECTIONS c (chemistry) AND h (electricity), fixed a priori for distinct
# vocabulary and ample rows (101k and 257k). Arms, as in the other split screens:
#   full        N from c + N from h
#   iid_half    a random N of those - IID denominator at matched size
#   expert_c    N from c
#   expert_h    N from h
#
# STREAMED, with a shuffle buffer, rather than downloaded: whole sections are
# gigabytes. Descriptions are truncated to a TOKEN budget with the Gemma
# tokeniser (patent text, like legislation, tokenises worse than prose), and any
# example whose rendered chat still exceeds sequence_len is dropped and counted.
#
# Usage, ON A LOGIN NODE:
#   uv run python prepare_bigpatent_data.py ../datasets/bigpatent
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

DATASET = "NortheasternUniversity/big_patent"
SECTIONS = {"c": "chemistry", "h": "electricity"}
INSTRUCTION = "Write the abstract for this patent description:"
SEQUENCE_LEN = 4096
DOCUMENT_TOKENS = 3300
TOKENIZER = "google/gemma-3-4b-it"


def convert_to_prompt(example):
    """One prompt; bigpatent_task.py imports this."""
    return {"messages": [{"role": "user", "content": f"{INSTRUCTION}\n\n{example['document']}"}]}


def to_training_example(example):
    return {"messages": convert_to_prompt(example)["messages"]
            + [{"role": "assistant", "content": example["summary"]}]}


def take(section, split, n, tokenizer, check_fit):
    stream = load_dataset(DATASET, section, split=split, streaming=True).shuffle(seed=42, buffer_size=5000)
    rows, dropped = [], 0
    for row in stream:
        ids = tokenizer(row["description"], add_special_tokens=False)["input_ids"]
        example = {"document": tokenizer.decode(ids[:DOCUMENT_TOKENS], skip_special_tokens=True),
                   "summary": row["abstract"].strip(), "shard": section,
                   "domain": SECTIONS[section]}
        if check_fit:
            rendered = tokenizer.apply_chat_template(to_training_example(example)["messages"], tokenize=False)
            if len(tokenizer(rendered)["input_ids"]) > SEQUENCE_LEN:
                dropped += 1
                continue
        rows.append(example)
        if len(rows) == n:
            break
    print(f"  {section} {split}: {len(rows)} rows ({dropped} dropped as over {SEQUENCE_LEN} tokens)")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare BigPatent for a domain-split screen.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-section", type=int, default=2000)
    parser.add_argument("--test-per-section", type=int, default=250)
    parser.add_argument("--validation-per-section", type=int, default=125)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    train = {s: take(s, "train", args.per_section, tokenizer, True) for s in SECTIONS}
    val = concatenate_datasets([take(s, "validation", args.validation_per_section, tokenizer, True)
                                for s in SECTIONS]).map(to_training_example)
    test = concatenate_datasets([take(s, "test", args.test_per_section, tokenizer, False)
                                 for s in SECTIONS])

    full = concatenate_datasets(list(train.values())).shuffle(seed=42)
    arms = {"full": full, "iid_half": full.select(range(len(full) // 2)),
            "expert_c": train["c"], "expert_h": train["h"]}
    for name, rows in arms.items():
        DatasetDict({"train": rows.map(to_training_example), "validation": val}).save_to_disk(
            str(args.output_dir / name))
        print(f"{name:>10}: {len(rows)} rows  c/h {sum(x == 'c' for x in rows['shard'])}/"
              f"{sum(x == 'h' for x in rows['shard'])}")
    (args.output_dir / "ARMS").write_text("\n".join(arms) + "\n")
    test.save_to_disk(str(args.output_dir / "test"))
    ref = sorted(len(tokenizer(s)["input_ids"]) for s in test["summary"])
    print(f"{'test':>10}: {len(test)} rows; abstract tokens p50 {ref[len(ref) // 2]}, "
          f"p99 {ref[int(.99 * len(ref))]}, max {ref[-1]} - bigpatent_task.py max_tokens must clear it")


if __name__ == "__main__":
    main()
