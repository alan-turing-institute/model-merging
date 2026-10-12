# Build English->Chinese domain-adaptation data for a merging headroom screen:
# UN (legal/diplomatic) against news commentary. The EN-DE medical/legal screen
# rebuilt for Chinese, where Gemma 3 4B is expected (not yet measured) to have
# more headroom than into German.
#
# SOURCES, and one trap. News training data is News Commentary, NOT wmt19's
# zh-en training set: that set is a mix that itself contains the UN corpus, so a
# "news" expert trained on it would partly be a UN expert. Tests:
#   un    held out from the UN stream, disjoint from training by English text
#   news  newstest2018 (wmt19 validation), professionally translated news
#
# UN BOILERPLATE. The UN corpus is dominated by headers and references
# ("UNITED NATIONS", "RESOLUTION 918 (1994)"). Pairs are kept only if the English
# has at least 6 words and is not mostly upper case, and are deduplicated on the
# English side. The same filter is applied to the held-out UN test - otherwise
# the test would score the model on copying document headers.
#
# Usage, ON A LOGIN NODE:
#   uv run python prepare_mt_zh_data.py ../datasets/mt_zh
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

INSTRUCTION = "Translate this English text into Chinese:"
SEQUENCE_LEN = 1024
TOKENIZER = "google/gemma-3-4b-it"


def convert_to_prompt(example):
    """One prompt; mt_zh_task.py imports this."""
    return {"messages": [{"role": "user", "content": f"{INSTRUCTION}\n\n{example['en']}"}]}


def to_training_example(example):
    return {"messages": convert_to_prompt(example)["messages"]
            + [{"role": "assistant", "content": example["zh"]}]}


def usable(en: str, zh: str) -> bool:
    letters = [c for c in en if c.isalpha()]
    return (len(en.split()) >= 6 and zh and zh != en
            and (not letters or sum(c.isupper() for c in letters) / len(letters) < 0.5))


def collect(stream, n, shard, tokenizer, exclude, label):
    rows, seen, scanned = [], set(exclude), 0
    for row in stream:
        scanned += 1
        en, zh = row["translation"]["en"].strip(), row["translation"]["zh"].strip()
        if en in seen or not usable(en, zh):
            continue
        seen.add(en)
        ex = {"en": en, "zh": zh, "domain": shard, "shard": shard}
        rendered = tokenizer.apply_chat_template(to_training_example(ex)["messages"], tokenize=False)
        if len(tokenizer(rendered)["input_ids"]) > SEQUENCE_LEN:
            continue
        rows.append(ex)
        if len(rows) == n:
            break
    print(f"  {label}: kept {len(rows)} of {scanned} scanned")
    if len(rows) < n:
        raise SystemExit(f"{label}: only {len(rows)} usable rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare EN-ZH UN/news MT for a domain-split screen.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-domain", type=int, default=4000)
    parser.add_argument("--test-per-domain", type=int, default=1000)
    parser.add_argument("--validation-per-domain", type=int, default=250)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    n_train, n_val, n_test = args.per_domain, args.validation_per_domain, args.test_per_domain

    # News: test is newstest2018; train/validation from News Commentary, with
    # anything in the test removed.
    news_test_ds = load_dataset("wmt/wmt19", "zh-en", split="validation").shuffle(seed=42)
    news_test = [{"en": r["translation"]["en"].strip(), "zh": r["translation"]["zh"].strip(),
                  "domain": "news", "shard": "news"} for r in news_test_ds.select(range(n_test))]
    nc = load_dataset("Helsinki-NLP/news_commentary", "en-zh", split="train").shuffle(seed=42)
    news = collect(nc, n_train + n_val, "news", tok, {r["en"] for r in news_test}, "news train+val")

    # UN: one shuffled stream; test first, then train and validation, so the
    # three are disjoint by construction (collect dedupes on English).
    un_stream = load_dataset("Helsinki-NLP/un_pc", "en-zh", split="train", streaming=True).shuffle(
        seed=42, buffer_size=50_000)
    un = collect(un_stream, n_test + n_train + n_val, "un", tok, set(), "un test+train+val")
    un_test, un_rest = un[:n_test], un[n_test:]

    train = {"un": Dataset.from_list(un_rest[:n_train]), "news": Dataset.from_list(news[:n_train])}
    val = Dataset.from_list(un_rest[n_train:] + news[n_train:]).map(to_training_example)
    full = concatenate_datasets(list(train.values())).shuffle(seed=42)
    arms = {"full": full, "iid_half": full.select(range(len(full) // 2)),
            "expert_un": train["un"], "expert_news": train["news"]}
    for name, rows in arms.items():
        DatasetDict({"train": rows.map(to_training_example), "validation": val}).save_to_disk(
            str(args.output_dir / name))
        print(f"{name:>12}: {len(rows)} rows  un/news {sum(x == 'un' for x in rows['shard'])}/"
              f"{sum(x == 'news' for x in rows['shard'])}")
    (args.output_dir / "ARMS").write_text("\n".join(arms) + "\n")
    test = Dataset.from_list(un_test + news_test)
    test.save_to_disk(str(args.output_dir / "test"))
    ref = sorted(len(tok(z)["input_ids"]) for z in test["zh"])
    print(f"{'test':>12}: {len(test)} rows; reference tokens p50 {ref[len(ref) // 2]}, "
          f"p99 {ref[int(.99 * len(ref))]}, max {ref[-1]}")


if __name__ == "__main__":
    main()
