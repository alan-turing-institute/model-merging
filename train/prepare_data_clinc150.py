# This file prepares CLINC150 intent classification for training, and for a
# domain-split merging experiment.
#
# Writes, each with a provenance.json (see ../provenance.py) that records the
# training prompt:
#   ../datasets/clinc150/clinc150_labels_dataset (--prompt labels, the default)
#   ../datasets/clinc150/clinc150_dataset        (--prompt short)
#       all 10 domains: train (15,000 queries, 100 per intent), validation (500)
#       and test (4,500, 30 per intent)
# and, only with --domain-split:
#   ../datasets/clinc150/split_2/clinc150_labels_domain_dataset_<i>_of_2
#   (clinc150_domain_dataset_<i>_of_2 with --prompt short)
#       the domain experts' data: the training queries of the five domains in
#       SHARDS[i - 1] (7,500), and the shared validation set
#
# Usage, from train/:
#   uv run python prepare_data_clinc150.py                 # the full dataset, labels prompt
#   uv run python prepare_data_clinc150.py --prompt short  # the full dataset, short prompt
#   uv run python prepare_data_clinc150.py --domain-split  # and the domain experts' data
# Random (IID) halves come from ../split_dataset.py, as for the other tasks:
#   python ../split_dataset.py 2 ../datasets/clinc150/clinc150_labels_dataset ../datasets/clinc150/split_2
#
# Why CLINC150: its 150 intents fall into 10 domains of exactly 15
# (clinc150_domains.json, from github.com/clinc/oos-eval, CC BY 3.0, Larson et
# al. 2019), a natural non-IID partition. Experts trained on different domains
# genuinely cannot do each other's work, which widens the gap a merge has to
# recover: (merged - better expert) / (full - better expert).
#
# What a domain split tests: with the short prompt, an expert never sees the
# other domains' label strings and cannot produce them, so most of its gap is
# label coverage. The labels prompt lists all 150 labels, which removes the
# coverage gap and leaves skill.
#
# Training prompt (--prompt): labels by default, so a trained model is scored
# on the prompt the untrained one can answer from (it cannot know the label
# strings otherwise), as crime's prompt names both of its answers. A labels
# example is ~650 tokens against ~35 for short, so training is much slower.
# clinc150_task.py scores either (-T prompt_variant=short|labels), which must
# match the prompt the model was trained on.
#
# Out-of-scope queries are dropped from every split: a domain expert would call
# every other domain out of scope, which contradicts the other expert once
# merged, and out-of-scope has no domain to split by.
import argparse
import json
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import provenance  # noqa: E402  (repo root, shared with split_dataset.py etc.)

SOURCE = "clinc/clinc_oos"
CONFIG = "plus"  # 100 training queries per intent
DATA_DIR = "../datasets/clinc150"
SPLIT_DIR = "../datasets/clinc150/split_2"
# The dataset name for each training prompt; short keeps the original name.
NAMES = {"labels": "clinc150_labels", "short": "clinc150"}
DOMAINS_FILE = Path(__file__).with_name("clinc150_domains.json")
VALIDATION_SIZE = 500

# Fixed before any result: transactional and out-of-home domains against
# household and assistant ones, five and five.
SHARDS = [
    ["banking", "credit_cards", "work", "travel", "auto_and_commute"],
    ["kitchen_and_dining", "home", "utility", "small_talk", "meta"],
]

INSTRUCTION = "Classify the user query into exactly one intent label. Answer with the label only."


def label_list():
    domains = json.loads(DOMAINS_FILE.read_text())
    return sorted(i for intents in domains.values() for i in intents)


# Both prompt variants for one query; clinc150_task.py imports this, so training
# and evaluation wording cannot drift apart.
def convert_to_prompt(example):
    query = example["text"]
    labels = ", ".join(label_list())
    return {
        "short": [{"role": "user", "content": f"{INSTRUCTION}\n\nQuery: {query}"}],
        "labels": [{"role": "user", "content": f"{INSTRUCTION}\n\nLabels: {labels}\n\nQuery: {query}"}],
    }


# The training format: the given prompt variant and the label as the answer.
def convert(example, prompt):
    return {"messages": convert_to_prompt(example)[prompt]
            + [{"role": "assistant", "content": example["label"]}]}


def parse_args():
    parser = argparse.ArgumentParser(description="Build CLINC150 intent classification prompts.")
    parser.add_argument(
        "--prompt", choices=sorted(NAMES), default="labels",
        help="The training prompt: labels lists all 150 labels, short does not "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--domain-split", action="store_true",
        help=f"Also write the two domain experts' training data under {SPLIT_DIR} "
             "(default: the full dataset only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    domains = json.loads(DOMAINS_FILE.read_text())
    assert sorted(domains) == sorted(SHARDS[0] + SHARDS[1]), "the shards must cover the 10 domains exactly"
    domain_of = {intent: d for d, intents in domains.items() for intent in intents}
    shard_of = {d: i + 1 for i, shard in enumerate(SHARDS) for d in shard}

    ds = load_dataset(SOURCE, CONFIG)
    names = ds["train"].features["intent"].names

    def annotate(split):
        # Each split is stored in blocks of one intent, so shuffle before taking
        # any subset, or a subset is a set of whole intents.
        rows = split.map(lambda r: {"label": names[r["intent"]]})
        rows = rows.filter(lambda r: r["label"] != "oos")
        rows = rows.map(lambda r: {"domain": domain_of[r["label"]], "shard": shard_of[domain_of[r["label"]]]})
        return rows.shuffle(seed=42)

    name = NAMES[args.prompt]
    output_dir = f"{DATA_DIR}/{name}_dataset"
    dataset = DatasetDict({
        "train": annotate(ds["train"]),
        "validation": annotate(ds["validation"]).select(range(VALIDATION_SIZE)),
        "test": annotate(ds["test"]),
    }).map(convert, fn_kwargs={"prompt": args.prompt})
    dataset.save_to_disk(output_dir)
    provenance.write(
        output_dir, __file__, "full", dataset,
        source=f"{SOURCE} ({CONFIG})",
        sampling=f"in-scope queries only; all of train and test, {VALIDATION_SIZE} of validation; shuffled, seed 42",
        prompt=args.prompt,
    )
    print(f"{output_dir}: {len(dataset['train'])} training queries from all 10 domains, {args.prompt} prompt")

    if not args.domain_split:
        return
    for i, shard in enumerate(SHARDS, start=1):
        part = DatasetDict({
            "train": dataset["train"].filter(lambda r: r["shard"] == i),
            "validation": dataset["validation"],
        })
        out = f"{SPLIT_DIR}/{name}_domain_dataset_{i}_of_2"
        part.save_to_disk(out)
        provenance.write(
            out, __file__, f"{i} of 2, by domain", part,
            source=output_dir, domains=shard,
            sampling="the training queries of these domains; the full validation set; no test split",
            prompt=args.prompt,
        )
        print(f"{out}: {len(part['train'])} training queries from {', '.join(shard)}")


if __name__ == "__main__":
    main()
