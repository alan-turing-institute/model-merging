# Build English->German domain-adaptation data for a merging headroom screen:
# medical (EMEA) against legal (JRC-Acquis), the Aharoni & Goldberg (2020)
# multi-domain splits as mirrored on the Hub.
#
# WHY THIS PAIR. Both experts do the SAME task - translate English into German -
# so a domain split here tests whether a merge combines domain SKILL, not label
# coverage. That is the property CLINC150's domain split lacks, and the reason to
# screen this alongside it. Same language pair also means neither expert is
# learning a language the base model cannot already write.
#
# ARMS, as in prepare_clinc150_data.py:
#   full            N medical + N legal
#   iid_half        a random N of those - the IID denominator, at matched size
#   expert_medical  N medical
#   expert_law      N legal
# N defaults to 4,000 per domain. The published domain curves for MT are long,
# so this is a deliberately small operating point; raise --per-domain if the
# screen shows the experts still climbing.
#
# DEDUPLICATION IS NOT OPTIONAL. EMEA is notorious for repeated boilerplate
# (dosage lines, section headings). Train rows are deduplicated on the English
# side, and any train row whose English appears in either domain's dev or test
# split is removed - otherwise a test score partly measures memorisation.
#
# LICENCES. The mirrors carry no licence tag; the terms are the sources' (the
# EMA's reuse notice for EMEA, the EU's reuse policy for JRC-Acquis). Check them
# before anything reaches a publication.
#
# Usage, ON A LOGIN NODE:
#   uv run python prepare_mt_domain_data.py ../datasets/mt_domain
import argparse
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

SOURCES = {
    "medical": "ahazeemi/opus-medical-en-de",
    "law": "ahazeemi/opus-law-en-de-new",
}
INSTRUCTION = "Translate this English text into German:"
SEQUENCE_LEN = 1024
TOKENIZER = "google/gemma-3-4b-it"


def convert_to_prompt(example):
    """One prompt; mt_domain_task.py imports this so wording cannot drift."""
    return {"messages": [{"role": "user", "content": f"{INSTRUCTION}\n\n{example['en']}"}]}


def to_training_example(example):
    return {"messages": convert_to_prompt(example)["messages"]
            + [{"role": "assistant", "content": example["de"]}]}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare EN-DE medical/legal MT for a domain-split screen.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-domain", type=int, default=4000, help="training rows per domain")
    parser.add_argument("--test-per-domain", type=int, default=1000)
    parser.add_argument("--validation-per-domain", type=int, default=250)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    n_tokens = lambda row: len(tokenizer.apply_chat_template(to_training_example(row)["messages"], tokenize=True))

    raw = {d: load_dataset(ident) for d, ident in SOURCES.items()}
    held_out = {r["en"].strip() for d in raw.values() for s in ("dev", "test") for r in d[s]}

    train, validation, test = {}, [], []
    for domain, ds in raw.items():
        seen, keep = set(held_out), []
        pool = ds["train"].shuffle(seed=42)
        for i, row in enumerate(pool):
            en, de = row["en"].strip(), row["de"].strip()
            if not en or not de or en in seen:
                continue
            seen.add(en)
            if n_tokens(row) > SEQUENCE_LEN:
                continue
            keep.append(i)
            if len(keep) == args.per_domain:
                break
        if len(keep) < args.per_domain:
            raise SystemExit(f"{domain}: only {len(keep)} usable rows after dedup and length filter")
        train[domain] = pool.select(keep).map(lambda r: {"domain": domain, "shard": domain})
        print(f"{domain:>8}: kept {len(keep)} of the first {i + 1} shuffled train rows "
              f"(the rest duplicates, held-out overlaps or over {SEQUENCE_LEN} tokens)")
        validation.append(ds["dev"].shuffle(seed=42).select(range(args.validation_per_domain))
                          .map(lambda r: {"domain": domain, "shard": domain}))
        test.append(ds["test"].shuffle(seed=42).select(range(args.test_per_domain))
                    .map(lambda r: {"domain": domain, "shard": domain}))

    full = concatenate_datasets(list(train.values())).shuffle(seed=42)
    arms = {
        "full": full,
        "iid_half": full.select(range(len(full) // 2)),
        "expert_medical": train["medical"],
        "expert_law": train["law"],
    }
    val = concatenate_datasets(validation).map(to_training_example)
    for name, rows in arms.items():
        DatasetDict({"train": rows.map(to_training_example), "validation": val}).save_to_disk(
            str(args.output_dir / name))
        mix = {d: sum(1 for x in rows["domain"] if x == d) for d in SOURCES}
        print(f"{name:>15}: {len(rows)} rows  medical/law {mix['medical']}/{mix['law']}")
    (args.output_dir / "ARMS").write_text("\n".join(arms) + "\n")

    test = concatenate_datasets(test)
    test.save_to_disk(str(args.output_dir / "test"))
    ref_tokens = sorted(len(tokenizer(r["de"])["input_ids"]) for r in test)
    print(f"{'test':>15}: {len(test)} rows; reference tokens p50 {ref_tokens[len(ref_tokens) // 2]}, "
          f"p99 {ref_tokens[int(.99 * len(ref_tokens))]}, max {ref_tokens[-1]} "
          f"- mt_domain_task.py's max_tokens must clear the max")


if __name__ == "__main__":
    main()
