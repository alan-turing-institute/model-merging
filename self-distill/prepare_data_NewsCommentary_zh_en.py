# This file prepares the WMT News Commentary zh-en corpus for self-distillation
# training on Chinese-to-English translation. It loads the corpus, cleans it,
# splits it by article, samples down to a manageable size, and builds three
# prompt variants per sentence pair.
#
# Source: WMT's own release, data.statmt.org/news-commentary/v18 (Project
# Syndicate commentaries, professionally translated). NOT the OPUS copy of
# News-Commentary zh-en: there, ~89% of links pair two English sentences with a
# Chinese paragraph that holds a third sentence the English side lacks, so a
# gold reference would teach the model to drop content. The WMT release is
# aligned one sentence to one sentence.
#
# The file is a two-column TSV (English, Chinese). A line that is blank on both
# sides separates two articles, and the line after it is the next article's
# title. We split train/validation/test by ARTICLE, so no test sentence shares
# an article with a training sentence. Each held-out pool is --holdout-factor
# times its sample size, so a 1,000-row test set spans ~250 articles rather than
# the ~30 it would take to hold out exactly 1,000 pairs.
#
# Cleaning, in order (counts are for v18, 373,736 lines):
#   - blank-on-one-side pairs (814)
#   - Chinese side with no Chinese characters (4,087): untranslated Project
#     Syndicate subscription boilerplate, e.g. "For a limited time, you can
#     gain greater access to Project Syndicate ..."
#   - exact duplicate pairs, keeping the first (~300 once the boilerplate is gone)
#   - English side under --min-words words (titles and fragments)
#   - English side over --max-words words (a few lines run to 2k+ tokens)
#   - Chinese characters per English word outside [--min-ratio, --max-ratio].
#     Below ~1 the Chinese has usually dropped a sentence; only CJK characters
#     count, because the Chinese often keeps English names in parentheses.
#
# As with XSum, the teacher and the student are deliberately given DIFFERENT
# prompts: the teacher a strict system prompt, the student a plainer one it has
# to internalise. With an identical prompt on both sides, self-distillation sits
# at a fixed point and teaches nothing. Unlike the XSum teacher prompt, these
# have not been through a bake-off; check a sample of teacher outputs before a
# full run.
#
# Fields emitted, alongside id, document_id, zh and en:
#   teacher_messages - what the teacher generates from (strict, system turn)
#   messages_long    - student prompt, instruction-heavy
#   messages_short   - student prompt, terse; the intended end state
#   reference        - the gold English translation, under the dataset-
#                      independent name that generate_teacher_logprobs.py
#                      --targets gold reads
#
# English references are short: mean 28 Gemma 3 tokens, p99.9 87, so
# generate_teacher_logprobs.py's default --max-tokens 128 drops almost nothing.
#
# A provenance.json (see ../provenance.py) records the source, arguments and git
# commit, with coverage "full": split_dataset.py pieces of it say "i of n".
#
# Usage:
#   python prepare_data_NewsCommentary_zh_en.py \
#       https://data.statmt.org/news-commentary/v18/training/news-commentary-v18.en-zh.tsv.gz \
#       ../datasets/translation/news_commentary_zh_en_prompts
#   python prepare_data_NewsCommentary_zh_en.py path/to/news-commentary-v18.en-zh.tsv.gz out_dir
import argparse
import gzip
import io
import random
import re
import sys
import urllib.request
from pathlib import Path

from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import provenance  # noqa: E402  (repo root, shared with split_dataset.py etc.)

SEED = 42

TEACHER_SYSTEM = (
    "You are a professional Chinese-to-English translator for an international "
    "opinion and commentary publication. You translate the Chinese text you are given "
    "into fluent, natural English that keeps every piece of information and the "
    "author's meaning, tone and register, adding nothing and leaving nothing out. "
    "You reply in plain text with the English translation only: no notes, no "
    "explanations, no pinyin, no quotation marks around it, and you never repeat "
    "the Chinese. Your entire reply is the translation itself."
)
# The ask goes AFTER the text so the most recent tokens are the instruction.
TEACHER_ASK = "Translate the Chinese text above into English."

STUDENT_LONG = (
    "Translate the following Chinese sentence from a news commentary into fluent,"
    " accurate English, keeping all of its meaning and adding nothing."
    "\n\nOutput only the translation."
)
STUDENT_SHORT = "Translate into English:"

CJK = re.compile(r"[一-鿿]")


def convert_to_prompt(example):
    source = example["zh"]
    return {
        "teacher_messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"Chinese:\n\n{source}\n\n{TEACHER_ASK}"},
        ],
        "messages_long": [
            {"role": "user", "content": f"{STUDENT_LONG}\n\nChinese:\n\n{source}"},
        ],
        "messages_short": [
            {"role": "user", "content": f"{STUDENT_SHORT}\n\n{source}"},
        ],
        "reference": example["en"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build zh-en translation self-distillation prompts from WMT News Commentary."
    )
    parser.add_argument(
        "dataset",
        help="The WMT news-commentary en-zh TSV (.tsv or .tsv.gz): a local path or a URL.",
    )
    parser.add_argument("output_dir", type=Path, help="Directory to save the prompts dataset into.")
    # Same sizes as the XSum split. Validation and test are drawn from whole
    # held-out articles; train is sampled from the remaining articles.
    parser.add_argument("--train-size", type=int, default=10_000, help="Rows to sample for train.")
    parser.add_argument("--validation-size", type=int, default=1_000, help="Rows to sample for validation.")
    parser.add_argument("--test-size", type=int, default=1_000, help="Rows to sample for test.")
    parser.add_argument(
        "--holdout-factor", type=int, default=10,
        help="Hold out articles holding this many times the validation/test size, then "
        "sample from them, for topic coverage (default: %(default)s).",
    )
    parser.add_argument("--min-words", type=int, default=4, help="Minimum English words per pair.")
    parser.add_argument("--max-words", type=int, default=100, help="Maximum English words per pair.")
    parser.add_argument(
        "--min-ratio", type=float, default=1.0,
        help="Minimum Chinese characters per English word (default: %(default)s).",
    )
    parser.add_argument(
        "--max-ratio", type=float, default=4.0,
        help="Maximum Chinese characters per English word (default: %(default)s).",
    )
    return parser.parse_args()


def read_lines(dataset: str):
    if dataset.startswith(("http://", "https://")):
        with urllib.request.urlopen(dataset) as response:
            raw = response.read()
    else:
        raw = Path(dataset).read_bytes()
    if dataset.endswith(".gz"):
        raw = gzip.decompress(raw)
    return io.StringIO(raw.decode("utf-8")).read().splitlines()


def load_articles(dataset: str, args):
    """The cleaned corpus as a list of articles, each a list of (en, zh) pairs."""
    articles = [[]]
    seen = set()
    dropped = {"one_side_blank": 0, "no_chinese": 0, "duplicate": 0, "length": 0, "ratio": 0}
    for line in read_lines(dataset):
        en, _, zh = line.partition("\t")
        en, zh = en.strip(), zh.strip()
        if not en and not zh:
            articles.append([])
            continue
        if not en or not zh:
            dropped["one_side_blank"] += 1
            continue
        n_cjk = len(CJK.findall(zh))
        if n_cjk == 0:
            dropped["no_chinese"] += 1
            continue
        if (en, zh) in seen:
            dropped["duplicate"] += 1
            continue
        seen.add((en, zh))
        n_words = len(en.split())
        if not args.min_words <= n_words <= args.max_words:
            dropped["length"] += 1
            continue
        if not args.min_ratio <= n_cjk / n_words <= args.max_ratio:
            dropped["ratio"] += 1
            continue
        articles[-1].append((en, zh))
    articles = [a for a in articles if a]
    n_pairs = sum(len(a) for a in articles)
    print(f"kept {n_pairs} pairs in {len(articles)} articles (dropped: {dropped})")
    return articles


def rows_for(articles, article_ids):
    return [
        {"id": f"{doc}-{i}", "document_id": doc, "en": en, "zh": zh}
        for doc in article_ids
        for i, (en, zh) in enumerate(articles[doc])
    ]


def main():
    args = parse_args()
    articles = load_articles(args.dataset, args)

    # Hold out whole articles for test, then validation, until each pool has
    # holdout_factor times the pairs it needs; everything left is the train pool.
    order = list(range(len(articles)))
    random.Random(SEED).shuffle(order)
    pools = {}
    cursor = 0
    for split, n in [("test", args.test_size), ("validation", args.validation_size)]:
        start = cursor
        count = 0
        while count < n * args.holdout_factor and cursor < len(order):
            count += len(articles[order[cursor]])
            cursor += 1
        pools[split] = order[start:cursor]
    pools["train"] = order[cursor:]

    sample_sizes = {"train": args.train_size, "validation": args.validation_size, "test": args.test_size}
    splits = {}
    for split, n in sample_sizes.items():
        rows = rows_for(articles, pools[split])
        if n > len(rows):
            raise ValueError(f"Requested {n} rows for {split!r}, but its articles only have {len(rows)}")
        splits[split] = Dataset.from_list(rows).shuffle(seed=SEED).select(range(n))

    # generate_teacher_logprobs.py loads this to run teacher generation.
    prompts_ds = DatasetDict(splits).map(convert_to_prompt)
    prompts_ds.save_to_disk(str(args.output_dir))
    provenance.write(
        args.output_dir, __file__, "full", prompts_ds, args,
        source=args.dataset,
        sampling=f"cleaned corpus split by article (seed {SEED}), then a seed-{SEED} "
                 "sample of --train-size/--validation-size/--test-size rows per split",
    )
    print(prompts_ds)


if __name__ == "__main__":
    main()
