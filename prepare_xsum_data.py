# Prepares the XSum summarisation dataset for the disjoint-halves merging
# experiment. Same shape as prepare_data.py (the crime classification arm):
# build one full dataset, then split its train/validation into two disjoint
# halves so that half 1 + half 2 == full, with no overlap.
#
# XSum (BBC News articles paired with the single-sentence summary the BBC
# itself wrote as the article's intro) is much bigger than the crime dataset -
# 204k train / 11k test - so everything is subsampled to keep a full
# train -> merge -> evaluate cycle to a few hours rather than a few days. The
# sizes are the knobs most worth turning, so they're env vars:
#
#   XSUM_TRAIN_N        examples in the full training set (halves get N/2 each)
#   XSUM_VAL_N          examples in the full validation set
#   XSUM_TEST_N         examples in the held-out test set (all variants share it)
#   XSUM_MAX_DOC_WORDS  articles are truncated to this many words
#   XSUM_SEED           subsample/split seed
#
# Truncation matters for cost: XSum articles run to several thousand words in
# the tail, and axolotl's sequence_len is a hard cap that silently drops
# whatever overflows. Cutting at 512 words keeps essentially every example
# inside the 2048-token window with room for the prompt and the summary, so
# training never sees an example whose target was truncated away. XSum's
# summary is the article's opening sentence, so the information needed is at
# the top anyway - this is much less lossy than it would be on a dataset with
# a genuinely dispersed summary.
import os

from datasets import DatasetDict, load_dataset

TRAIN_N = int(os.environ.get("XSUM_TRAIN_N", 8000))
VAL_N = int(os.environ.get("XSUM_VAL_N", 500))
TEST_N = int(os.environ.get("XSUM_TEST_N", 1000))
MAX_DOC_WORDS = int(os.environ.get("XSUM_MAX_DOC_WORDS", 512))
SEED = int(os.environ.get("XSUM_SEED", 42))

INSTRUCTION = (
    "Summarise the following BBC News article in a single short sentence. "
    "Give only the summary sentence, with no preamble.\n\n"
)

# Loading note: EdinburghNLP/xsum is served as parquet on the Hub, so no
# trust_remote_code is needed with a current `datasets`. On older versions that
# still resolve the loading script you would need trust_remote_code=True.
ds = load_dataset("EdinburghNLP/xsum")


def subsample(split, n):
    """Deterministic n-example subsample, or the whole split if it's smaller."""
    split = split.shuffle(seed=SEED)
    return split.select(range(min(n, len(split))))


def convert(example):
    """Render the prompt once, here, and carry it in the dataset.

    Training and evaluation must see byte-identical prompts; the crime arm
    keeps the wording in two places (prepare_data.py and evaluate.py) and
    relies on them being kept in sync by hand. Storing the rendered prompt as a
    column removes that failure mode - evaluate_summarisation.py reads this
    column rather than rebuilding the string.
    """
    document = " ".join(example["document"].split()[:MAX_DOC_WORDS])
    summary = example["summary"].strip()
    prompt = INSTRUCTION + document
    return {
        "document": document,
        "summary": summary,
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": summary},
        ],
    }


dataset = DatasetDict(
    {
        "train": subsample(ds["train"], TRAIN_N),
        "validation": subsample(ds["validation"], VAL_N),
        "test": subsample(ds["test"], TEST_N),
    }
).map(convert)

dataset.save_to_disk("../datasets/xsum_dataset")

# --- the two disjoint halves ---
#
# No stratification here, unlike the crime arm: there's no label to stratify
# on. The halves are a plain random split of the already-shuffled subsample,
# which makes them exchangeable draws from the same distribution - which is
# exactly the assumption the experiment tests.

train_split = dataset["train"].train_test_split(test_size=0.5, seed=SEED)
val_split = dataset["validation"].train_test_split(test_size=0.5, seed=SEED)

DatasetDict(
    {"train": train_split["train"], "validation": val_split["train"]}
).save_to_disk("../datasets/xsum_dataset1")

DatasetDict(
    {"train": train_split["test"], "validation": val_split["test"]}
).save_to_disk("../datasets/xsum_dataset2")

print(
    f"full   train={len(dataset['train'])} "
    f"validation={len(dataset['validation'])} test={len(dataset['test'])}"
)
print(
    f"half 1 train={len(train_split['train'])} "
    f"validation={len(val_split['train'])}"
)
print(
    f"half 2 train={len(train_split['test'])} "
    f"validation={len(val_split['test'])}"
)
