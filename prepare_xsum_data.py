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
from pathlib import Path

from datasets import DatasetDict, load_dataset

TRAIN_N = int(os.environ.get("XSUM_TRAIN_N", 8000))
VAL_N = int(os.environ.get("XSUM_VAL_N", 500))
TEST_N = int(os.environ.get("XSUM_TEST_N", 1000))
MAX_DOC_WORDS = int(os.environ.get("XSUM_MAX_DOC_WORDS", 512))
SEED = int(os.environ.get("XSUM_SEED", 42))
# Number of disjoint shards the training set is split into. 2 reproduces the
# halves design; 8 is the small-shard variant, which exists because the halves
# arm saturated - 4,000 XSum examples already reach 0.3990 against full-data's
# 0.4049, leaving a 0.0059 gap that a paired bootstrap cannot resolve at
# n = 1000. Smaller shards make weaker experts and so a larger gap for the
# recovery fraction to normalise by.
SPLITS = int(os.environ.get("XSUM_SPLITS", 2))
# "plain" (default) reproduces the arm every number on record was measured on:
# one prompt, used by the teacher, the student and the evaluator alike. "ctx"
# builds the context-distillation variant, where the teacher is conditioned on a
# strict system prompt and the student sees a terse one - the teacher's edge is
# then partly context rather than only weights. It writes to its own tree so the
# two arms cannot contaminate each other.
VARIANT = os.environ.get("XSUM_VARIANT", "plain")
if VARIANT not in ("plain", "ctx"):
    raise SystemExit(f"XSUM_VARIANT must be plain or ctx, got {VARIANT!r}")
PREFIX = "xsum_dataset" if VARIANT == "plain" else "xsum_ctx_dataset"

# --- the context-distillation prompts (XSUM_VARIANT=ctx) ---
#
# Adapted from jmcinroy's self-distill arm on `main`, where a four-way bake-off
# scored this system prompt at 92.5% clean output - no preamble, no markdown,
# one sentence of 20-25 words - against 59.5% for an instruction carried in the
# user turn, which is the shape INSTRUCTION below uses. Few-shot variants scored
# worse and cost 2.3x the context, so there are deliberately no exemplars.
#
# Constraints are prose, not a bulleted "Rules:" block: an earlier version of
# theirs echoed the literal label into ~6% of outputs.
TEACHER_SYSTEM = (
    "You are a BBC News sub-editor. You reply with exactly one sentence of 20 to 25 "
    "words, written in the present tense and the third person, in plain text with no "
    "markdown, no asterisks and no quotation marks. You never introduce, label or "
    "explain the sentence, and you never restate the request. Your entire reply is "
    "the sentence itself."
)
# The ask goes AFTER the article, so the most recent tokens are the instruction
# rather than several hundred words of news copy.
TEACHER_ASK = "Summarise the article above in one sentence of 20 to 25 words."
# What the student sees. Terse by design: the point of the arm is that the
# format has to end up in the weights rather than be re-read every time.
STUDENT_INSTRUCTION = "Summarise this BBC News article in one sentence:\n\n"

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
    prompt = (STUDENT_INSTRUCTION if VARIANT == "ctx" else INSTRUCTION) + document
    record = {
        "document": document,
        "summary": summary,
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": summary},
        ],
    }
    if VARIANT == "ctx":
        # The teacher's view. Same assistant target - that is what makes the
        # teacher's distributions transplantable onto the student's sequence -
        # but a different context, which is the whole point of the arm.
        record["teacher_messages"] = [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"Article:\n\n{document}\n\n{TEACHER_ASK}"},
            {"role": "assistant", "content": summary},
        ]
    return record


dataset = DatasetDict(
    {
        "train": subsample(ds["train"], TRAIN_N),
        "validation": subsample(ds["validation"], VAL_N),
        "test": subsample(ds["test"], TEST_N),
    }
).map(convert)

dataset.save_to_disk(f"../datasets/{PREFIX}")

# --- the two disjoint halves ---
#
# No stratification here, unlike the crime arm: there's no label to stratify
# on. The halves are a plain random split of the already-shuffled subsample,
# which makes them exchangeable draws from the same distribution - which is
# exactly the assumption the experiment tests.

# SPLITS == 2 MUST keep the original construction. train_test_split shuffles
# again with its own seed before cutting, so a contiguous shard assigns
# different examples to half 1 than the call below does - and every half-expert
# adapter, merge and result on record was trained against the train_test_split
# membership. Reproducing it is not a style preference; a contiguous rewrite
# would silently make new halves incomparable with every existing number.
if SPLITS == 2:
    _train = dataset["train"].train_test_split(test_size=0.5, seed=SEED)
    _val = dataset["validation"].train_test_split(test_size=0.5, seed=SEED)
    _members = [
        ({"train": _train["train"], "validation": _val["train"]}),
        ({"train": _train["test"], "validation": _val["test"]}),
    ]

    def shard(split, index):
        key = "train" if split is dataset["train"] else "validation"
        return _members[index][key]
else:
    # The subsample above is already shuffled with SEED, so contiguous slices
    # are exchangeable draws, and a plain slice makes the disjointness obvious
    # by construction: shard i and shard j share no index, and the union is the
    # whole split.
    def shard(split, index):
        return split.shard(num_shards=SPLITS, index=index, contiguous=True)


# The shard directories are named xsum_dataset1..N for every N, so a run at a
# different SPLITS overwrites the previous one in place. That is not a cosmetic
# collision: xsum_dataset1 and xsum_dataset2 hold the two-way halves that every
# adapter, merge and KD number on record was trained against, and an
# XSUM_SPLITS=8 run would replace half 1 with a 1/8 shard under the same path,
# leaving nothing to notice. Each directory records the SPLITS it was built
# with; a run that would change that refuses unless XSUM_OVERWRITE=1.
for i in range(SPLITS):
    out = Path(f"../datasets/{PREFIX}{i + 1}")
    stamp = out / ".xsum_splits"
    if out.is_dir() and os.environ.get("XSUM_OVERWRITE") != "1":
        previous = stamp.read_text().strip() if stamp.is_file() else None
        if previous != str(SPLITS):
            built = f"XSUM_SPLITS={previous}" if previous else "an unrecorded XSUM_SPLITS"
            raise SystemExit(
                f"{out} already exists and was built with {built}; this run is "
                f"XSUM_SPLITS={SPLITS}. Overwriting it would silently change what "
                f"every artefact trained on that path was trained on - the halves "
                f"at xsum_dataset1/2 are what every adapter, merge and KD number "
                f"on record used. Move the old datasets aside, or set "
                f"XSUM_OVERWRITE=1 if you really mean to replace them."
            )
    DatasetDict(
        {
            "train": shard(dataset["train"], i),
            "validation": shard(dataset["validation"], i),
        }
    ).save_to_disk(str(out))
    stamp.write_text(f"{SPLITS}\n")

print(
    f"full   train={len(dataset['train'])} "
    f"validation={len(dataset['validation'])} test={len(dataset['test'])}"
)
for i in range(SPLITS):
    print(
        f"shard {i + 1}/{SPLITS} "
        f"train={len(shard(dataset['train'], i))} "
        f"validation={len(shard(dataset['validation'], i))}"
    )
