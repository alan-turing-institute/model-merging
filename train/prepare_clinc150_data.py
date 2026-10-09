# Build CLINC150 training and evaluation data for a merging headroom screen.
#
# WHY CLINC150. It is the only classification set in the candidate list with a
# NATURAL non-IID partition: the authors group the 150 intents into 10 domains of
# exactly 15 (clinc150_domains.json, vendored from github.com/clinc/oos-eval,
# CC BY 3.0, Larson et al. 2019). Domain experts that genuinely cannot do each
# other's work are the strongest lever for widening the denominator of the
# recovery fraction, R = (merged - best expert) / (full - best expert).
#
# WHAT THE SCREEN MEASURES. Four trained arms plus the untrained base:
#   full       all 10 domains, 15,000 rows (100 per intent)
#   iid_half   a random 7,500 of those - the IID denominator, at matched size
#   expert_a   domains in SHARD_A, 7,500 rows
#   expert_b   domains in SHARD_B, 7,500 rows
# full - iid_half is the denominator a random split would give; full - best
# expert is the one a domain split gives. The screen reads both, so the case for
# the non-IID design is measured rather than asserted.
#
# BE CLEAR WHAT A DOMAIN SPLIT TESTS. Without the label list in the prompt, an
# expert has never seen the other shard's label strings and cannot produce them,
# so most of its gap to the full model is LABEL COVERAGE. A merge that recovers
# it is combining label vocabularies - a legitimate merging question, but a
# different one from combining skill on a shared task. The `labels` prompt
# variant lists all 150 labels, which removes the coverage gap and leaves skill;
# the screen trains on `short` and scores the base model on both.
#
# OUT-OF-SCOPE IS DROPPED from every split. A domain expert would reasonably call
# every other domain "out of scope", which contradicts the other expert's labels
# once merged; OOS has no domain to shard it by either.
#
# Usage, ON A LOGIN NODE:
#   uv run python prepare_clinc150_data.py ../datasets/clinc150
import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

DOMAINS_FILE = Path(__file__).with_name("clinc150_domains.json")

# Fixed a priori, before any result: transactional and out-of-home domains
# against household and assistant ones. Five and five, 7,500 rows each.
SHARD_A = ["banking", "credit_cards", "work", "travel", "auto_and_commute"]
SHARD_B = ["kitchen_and_dining", "home", "utility", "small_talk", "meta"]

INSTRUCTION = "Classify the user query into exactly one intent label. Answer with the label only."

SEQUENCE_LEN = 1024     # the `labels` variant is ~700 tokens; `short` ~60
TOKENIZER = "google/gemma-3-4b-it"


def label_list() -> list[str]:
    domains = json.loads(DOMAINS_FILE.read_text())
    return sorted(i for intents in domains.values() for i in intents)


def convert_to_prompt(example):
    """Prompt variants for one query. clinc150_task.py imports this."""
    query = example["text"]
    labels = ", ".join(label_list())
    return {
        "messages_short": [
            {"role": "user", "content": f"{INSTRUCTION}\n\nQuery: {query}"},
        ],
        "messages_labels": [
            {"role": "user", "content": f"{INSTRUCTION}\n\nLabels: {labels}\n\nQuery: {query}"},
        ],
    }


def to_training_example(example, prompt_variant: str):
    return {"messages": convert_to_prompt(example)[f"messages_{prompt_variant}"]
            + [{"role": "assistant", "content": example["label"]}]}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CLINC150 for a domain-split headroom screen.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", default="clinc/clinc_oos")
    parser.add_argument("--config", default="plus", help="plus has 100 training queries per intent")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--prompt-variant", default="short", choices=["short", "labels"])
    parser.add_argument("--tokenizer", default=TOKENIZER)
    return parser.parse_args()


def main():
    args = parse_args()
    domains = json.loads(DOMAINS_FILE.read_text())
    assert sorted(domains) == sorted(SHARD_A + SHARD_B), "shards must cover the 10 domains exactly"
    domain_of = {intent: d for d, intents in domains.items() for intent in intents}
    shard_of = {d: ("a" if d in SHARD_A else "b") for d in domains}

    ds = load_dataset(args.dataset, args.config)
    names = ds["train"].features["intent"].names
    missing = set(domain_of) - set(names)
    assert not missing, f"domain map names intents the dataset lacks: {sorted(missing)[:5]}"

    def annotate(split):
        # Every split is stored in contiguous label blocks, so shuffle before any
        # subsetting or a "random half" is really a set of whole intents.
        rows = split.map(lambda r: {"label": names[r["intent"]]})
        rows = rows.filter(lambda r: r["label"] != "oos")
        rows = rows.map(lambda r: {"domain": domain_of[r["label"]],
                                   "shard": shard_of[domain_of[r["label"]]]})
        return rows.shuffle(seed=42)

    train, validation, test = (annotate(ds[s]) for s in ("train", "validation", "test"))
    validation = validation.select(range(min(args.validation_size, len(validation))))

    arms = {
        "full": train,
        "iid_half": train.select(range(len(train) // 2)),
        "expert_a": train.filter(lambda r: r["shard"] == "a"),
        "expert_b": train.filter(lambda r: r["shard"] == "b"),
    }

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    render = lambda r: to_training_example(r, args.prompt_variant)
    val_rendered = validation.map(render)
    over = 0
    for name, rows in arms.items():
        arm = DatasetDict({"train": rows.map(render), "validation": val_rendered})
        arm.save_to_disk(str(args.output_dir / name))
        longest = max(len(tokenizer(tokenizer.apply_chat_template(m, tokenize=False))["input_ids"]) for m in arm["train"]["messages"])
        over += longest > SEQUENCE_LEN
        shards = {s: sum(1 for x in rows["shard"] if x == s) for s in ("a", "b")}
        print(f"{name:>10}: {len(rows)} rows  shard a/b {shards['a']}/{shards['b']}  longest {longest} tokens")

    # Largest first, matching the array index order in train_screen.sbatch.
    (args.output_dir / "ARMS").write_text("\n".join(arms) + "\n")
    test.save_to_disk(str(args.output_dir / "test"))
    print(f"{'test':>10}: {len(test)} in-scope rows (OOS dropped)")
    if over:
        raise SystemExit(f"examples exceed sequence_len {SEQUENCE_LEN}; raise it in clinc150_gemma.yaml too")


if __name__ == "__main__":
    main()
