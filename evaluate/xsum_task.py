"""XSum summarisation as an Inspect AI task.

The Inspect equivalent of old/evaluate_summarisation.py - ROUGE plus the semantic
and entity-grounding scores - expressed as a Task so it gets Inspect's log
format, `inspect view`, sample-level introspection and its own CLI:

    inspect eval xsum_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=short

A full model saved on disk (config.json + model.safetensors) loads with the
plain hf provider; the name after hf/ is only a label, model_path is what loads:

    inspect eval xsum_task.py --model hf/gemma3-xsum-self-dist-short \
      -M model_path=../models/gemma3-xsum-self-dist-short \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=short

evaluate.sbatch (Isambard-AI) and Azure/evaluate.sh build these commands.

PROMPT PARITY: the prompt is built here, from the article, by the same
`convert_to_prompt` that self-distill/prepare_data_XSum.py uses to build the
training data - imported, not copied, so training and evaluation wording cannot
drift apart. `prompt_variant` picks which of its prompts to evaluate with, and
it MUST match the one the model was trained on: a student trained on the short
prompt and scored on the long one is being measured on a task it never saw.
The variant is a task argument, so it is recorded in every .eval log.

The dataset therefore needs only `document` and `summary` (and ideally `id`):
datasets/xsum_prompts/test works as it is, as does any XSum split saved with
`save_to_disk`.

NO PREPARED DATA NEEDED. If the default split is missing, it is rebuilt from
the Hub - the same 1,000 test articles prepare_data_XSum.py selects
(shuffle(seed=42), first 1,000) at a pinned revision - and saved there. Compute
nodes have no network, so on Isambard build it once on a login node:

    uv run python xsum_task.py

DOCUMENT_WORDS, AND THE REPORT'S NUMBERS. The Isambard results in the report
were produced from prepare_xsum_data.py on the xsum-summarisation branch, which
collapsed whitespace and cut each article to 512 words (211 of the 1,000 test
articles are longer). prepare_data_XSum.py on main does neither. Pass
-T document_words=512 to reproduce those inputs exactly - verified identical
for all 1,000 rows against the split the report used. Like prompt_variant, it
must match what the model was trained on, and it is recorded in every log.
"""

import importlib.util
import math
import re
from pathlib import Path

from datasets import load_dataset, load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from rouge_score import rouge_scorer as rouge_scoring

from semantic_metrics import score_one, unsupported_entities
from xsum_faithfulness import faithfulness as _faithfulness

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

# Resolved from this file rather than the working directory, so the default
# holds wherever `inspect eval` is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "xsum_prompts" / "test"
PREPARE_SCRIPT = REPO_ROOT / "self-distill" / "prepare_data_XSum.py"

# Where the default split is rebuilt from when it is missing. The revision is
# pinned so the rebuild cannot drift if the Hub repo changes; at this revision
# shuffle(seed=42) + select(range(1000)) gives the same ids, in the same order,
# as the split the report's XSum results used.
XSUM_HUB = "EdinburghNLP/xsum"
XSUM_REVISION = "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9"
XSUM_TEST_SIZE = 1_000

# prompt_variant -> the field of convert_to_prompt's output holding that chat.
PROMPT_FIELDS = {
    "short": "messages_short",
    "long": "messages_long",
    "teacher": "teacher_messages",
}


def load_convert_to_prompt():
    """convert_to_prompt from self-distill/prepare_data_XSum.py.

    Loaded by path because self-distill/ is a directory of scripts, not an
    importable package.
    """
    spec = importlib.util.spec_from_file_location("prepare_data_XSum", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_to_prompt


# --- format checks ----------------------------------------------------------
#
# The self-distillation target is one sentence of 20-25 words with no preamble.
# These measure each of those directly, plus the word repetition that the KD
# alignment bug produced (self-distill/KD_ALIGNMENT_BUG.md).

TARGET_WORDS = (20, 25)

# Words that end in a full stop without ending a sentence. Compared lower-case,
# with the full stop removed.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "gov", "sen", "rep", "gen", "col",
    "lt", "sgt", "rev", "sir", "jr", "sr", "co", "corp", "inc", "ltd", "plc",
    "no", "vs", "etc", "approx", "dept", "est", "fig", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
# A single letter or dotted initials such as "J" or "U.S" (the final full stop
# already removed).
_INITIALS = re.compile(r"^(?:[A-Za-z]\.)*[A-Za-z]$")
_CLOSING = "\"'”’)]"


def count_sentences(text: str) -> int:
    """Number of sentences, not fooled by abbreviations or decimals.

    A sentence ends at a word ending in . ! or ? (before any closing quote or
    bracket), unless that word is an abbreviation or an initial. Decimals such
    as "4.2" never end in a full stop, so they are not counted. A final run of
    words with no closing punctuation - a truncated generation - counts as one
    more sentence. A sentence ending in "U.S." is under-counted; that is rare
    and errs towards one sentence, not away from it.
    """
    count = 0
    open_sentence = False
    for word in text.split():
        stripped = word.rstrip(_CLOSING)
        open_sentence = True
        if not stripped or stripped[-1] not in ".!?":
            continue
        if stripped[-1] == ".":
            body = stripped.rstrip(".")
            if body.lower() in _ABBREVIATIONS or _INITIALS.match(body):
                continue
        count += 1
        open_sentence = False
    return count + open_sentence


# Openings that introduce a summary rather than being one: "Here's a
# one-sentence summary of the BBC News article:", "Sure!", "**Summary:**".
_PREAMBLE_OPENING = re.compile(
    r"^\W*(?:here(?:'|’)?s\b|here is\b|sure\b|okay\b|certainly\b|of course\b|"
    r"summary\b|in one sentence\b|(?:my|a|the) (?:one[- ]sentence )?summary\b)",
    re.IGNORECASE,
)
# Any first line ending in a colon with more text after it - the shape every
# preamble in the base models' outputs takes, whatever its wording.
_PREAMBLE_LINE = re.compile(r"^[^\n]*:[*\s]*\n\s*\S")


# Openings that talk about the article instead of summarising it: "This BBC
# News article highlights...", "The article details...", "According to the BBC
# News article, ...". Only "article"-type nouns, and only followed by a
# reporting verb (or after "according to"), so "A UN report highlights..." or
# "A BBC Spotlight investigation revealed..." - real reports, real BBC
# programmes - are not caught.
_META_OPENING = re.compile(
    r"^\W*(?P<according>according to |in )?(?:this|the) "
    r"(?P<noun>(?:bbc(?: news)?(?: online)? |news |online )?(?:article|piece|text|passage))"
    r"(?(according)\b|,? (?:details|describes|highlights|reports|reflects|"
    r"introduces|promotes|summari[sz]es|offers|asks|discusses|explains|examines|"
    r"covers|focuses|reveals|tells|looks|explores|outlines|presents|provides|"
    r"features|announces|argues|states|says|notes|recounts|celebrates|profiles|"
    r"is|was)\b)",
    re.IGNORECASE,
)


def talks_about_article(text: str, document: str) -> bool:
    """Whether the output opens by describing the article rather than its news.

    Not counted when the source itself uses the same phrase - an article about
    a BBC News article can be summarised as "The BBC News article ..." - unless
    the phrase is only followed by a number there, as in "Article 50".
    """
    match = _META_OPENING.match(text)
    if not match:
        return False
    noun = r"\s+".join(map(re.escape, match["noun"].split()))
    return not re.search(rf"\b{noun}\b(?!\s*\d)", document, re.IGNORECASE)


def has_preamble(text: str, document: str = "") -> bool:
    return bool(
        _PREAMBLE_OPENING.match(text)
        or _PREAMBLE_LINE.match(text)
        or talks_about_article(text, document)
    )


# The same word twice in a row, separated only by whitespace, so "very, very"
# is not caught but "legal legal" is.
_REPEATED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
# The same word four or more times in a row: "family family family family".
_REPEATED_RUN = re.compile(r"\b(\w+)(?:\s+\1\b){3,}", re.IGNORECASE)


def repeated_words(prediction: str, document: str) -> list[str]:
    """Immediately repeated words in the prediction that the article does not
    repeat itself, so a name such as "Tian Tian" is not counted."""
    haystack = " ".join(document.split()).lower()
    return [
        m.group(0)
        for m in _REPEATED_WORD.finditer(prediction)
        if " ".join(m.group(0).split()).lower() not in haystack
    ]


def to_chat(messages):
    roles = {"system": ChatMessageSystem, "user": ChatMessageUser}
    return [roles[m["role"]](content=m["content"]) for m in messages]


def build_default_split():
    """Rebuild datasets/xsum_prompts/test from the Hub and save it.

    Same selection as prepare_data_XSum.py. Needs network: on Isambard, run it
    on a login node (`uv run python xsum_task.py`), not inside a job.
    """
    split = (load_dataset(XSUM_HUB, split="test", revision=XSUM_REVISION)
             .shuffle(seed=42).select(range(XSUM_TEST_SIZE)))
    split.save_to_disk(str(DEFAULT_DATASET))
    return split


def load_split(dataset_path):
    """The saved split at dataset_path; for the default path only, rebuilt from
    the Hub if it is missing. A wrong path passed explicitly fails loudly
    rather than silently scoring something else."""
    path = Path(dataset_path)
    if path.exists():
        return load_from_disk(str(path))
    if path != DEFAULT_DATASET:
        raise FileNotFoundError(f"no dataset at {path}")
    try:
        return build_default_split()
    except Exception as error:
        raise RuntimeError(
            f"{path} is missing and could not be rebuilt from the Hub ({error}). "
            "On a machine without network - an Isambard compute node - run "
            "`uv run python xsum_task.py` on a login node first."
        ) from error


def truncate_words(text, document_words):
    """Collapse whitespace and keep the first document_words words: exactly what
    prepare_xsum_data.py did to the articles behind the report's numbers."""
    return " ".join(text.split()[:document_words]) if document_words else text


def xsum_samples(dataset_path, prompt_variant, limit=None,
                 reference_column="summary", document_words=None):
    field = PROMPT_FIELDS[prompt_variant]
    convert_to_prompt = load_convert_to_prompt()
    dataset = load_split(dataset_path)
    if document_words:
        dataset = dataset.map(
            lambda row: {"document": truncate_words(row["document"], document_words)})
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        Sample(
            input=to_chat(convert_to_prompt(row)[field]),
            target=row[reference_column],
            id=row.get("id", index),
            # The source article, for entity_support. A reference-only scorer
            # cannot tell an invented name from a correct one - both are simply
            # absent from a 21-word reference - so grounding needs the document.
            metadata={"document": row["document"]},
        )
        for index, row in enumerate(dataset)
    ]


@scorer(
    metrics={
        "rouge1": [mean(), stderr()],
        "rouge2": [mean(), stderr()],
        "rougeL": [mean(), stderr()],
    }
)
def rouge():
    """Per-sample ROUGE F1 against the reference summary."""
    scoring = rouge_scoring.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        scores = scoring.score(reference, prediction)
        value = {rouge_type: scores[rouge_type].fmeasure for rouge_type in ROUGE_TYPES}
        return Score(
            value=value,
            answer=prediction,
            # Surfaced in `inspect view`, so a sample that looks wrong can be
            # read against its reference without leaving the viewer.
            explanation=f"reference: {reference}",
        )

    return score


@scorer(
    metrics={
        # Not quality measures - drift detectors. A merged model sliding back
        # towards base behaviour, or one task's output format leaking into
        # another, moves these before it moves ROUGE.
        "pred_words": [mean()],
        "pred_sentences": [mean()],
        # Shares of outputs, so the mean is the fraction meeting each check.
        "in_word_range": [mean(), stderr()],
        "one_sentence": [mean(), stderr()],
        "preamble": [mean(), stderr()],
        "repeated_word": [mean(), stderr()],
        "repeated_run": [mean()],
    }
)
def format_checks():
    """Whether each output has the trained format: one sentence, 20-25 words,
    no preamble, no repeated words.

    Counted on the whole completion, preamble included, since that is what the
    model produced.
    """
    low, high = TARGET_WORDS

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        words = len(prediction.split())
        sentences = count_sentences(prediction)
        document = (state.metadata or {}).get("document", "")
        repeats = repeated_words(prediction, document)
        value = {
            "pred_words": float(words),
            "pred_sentences": float(sentences),
            "in_word_range": float(low <= words <= high),
            "one_sentence": float(sentences == 1),
            "preamble": float(has_preamble(prediction, document)),
            "repeated_word": float(bool(repeats)),
            "repeated_run": float(bool(_REPEATED_RUN.search(prediction))),
        }
        return Score(
            value=value,
            answer=prediction,
            explanation=f"{words} words, {sentences} sentences"
            + (f" | repeated: {', '.join(repeats)}" if repeats else ""),
        )

    return score


# --- semantic scoring -------------------------------------------------------
#
# The metrics themselves live in semantic_metrics.py, shared with
# old/evaluate_summarisation.py. See that module for why ROUGE alone is a poor proxy
# on XSum in both directions, and for the measured hallucination case that
# motivates entity_support.


@scorer(
    metrics={
        "semantic": [mean(), stderr()],
        "entity_support": [mean(), stderr()],
        "entities_unsupported": [mean()],
        "entity_gradeable": [mean()],
    }
)
def semantic():
    """Embedding similarity to the reference, and entity grounding in the source."""

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        document = (state.metadata or {}).get("document", "")
        value = score_one(prediction, reference, document)
        bad = unsupported_entities(prediction, document)
        # score_one returns None for a non-empty summary that names nobody -
        # nothing to ground. NaN is Inspect's "unscored" marker for a key of a
        # dict-valued score: that sample is left out of entity_support's mean
        # and stderr, so the logged figure is the mean over gradeable summaries
        # only, as old/evaluate_summarisation.py computes it. entity_gradeable
        # keeps the denominator visible.
        value["entity_gradeable"] = 0.0 if value["entity_support"] is None else 1.0
        if value["entity_support"] is None:
            value["entity_support"] = math.nan
        return Score(
            value=value,
            answer=prediction,
            explanation=(
                f"reference: {reference}"
                + (f" | unsupported: {', '.join(bad)}" if bad else "")
            ),
        )

    return score


@task
def xsum(
    prompt_variant: str = "short",
    dataset_path: str = str(DEFAULT_DATASET),
    limit: int | None = None,
    reference_column: str = "summary",
    faithfulness: bool = False,
    document_words: int | None = None,
) -> Task:
    """XSum single-sentence summarisation, scored by ROUGE and grounding.

    prompt_variant: which prepare_data_XSum.py prompt to evaluate with - one of
    "short", "long" or "teacher". Must match what the model was trained on.

    faithfulness: also run the reference-free grounding measures, the G-Eval
    judge and UniEval's four dimensions (xsum_faithfulness.py). Off by default
    because it loads about 6 GB of extra models; pass -T faithfulness=true to
    enable. Scoring happens inline, so these land in the same .eval log as
    ROUGE rather than needing a separate re-scoring pass over stored JSON.

    document_words: collapse whitespace and cut each article to this many
    words before prompting. 512 reproduces the inputs behind the report's
    Isambard results; leave unset for prepare_data_XSum.py's full articles.
    Must match what the model was trained on.
    """
    if prompt_variant not in PROMPT_FIELDS:
        raise ValueError(
            f"prompt_variant must be one of {sorted(PROMPT_FIELDS)}, "
            f"got {prompt_variant!r}"
        )

    return Task(
        dataset=MemoryDataset(
            xsum_samples(dataset_path, prompt_variant, limit, reference_column,
                         document_words),
            name="xsum",
        ),
        solver=generate(),
        scorer=[rouge(), format_checks(), semantic()]
               + ([_faithfulness()] if faithfulness else []),
        # The models were trained earlier with a teacher which had 128 tokens.
        # XSum references average ~21 words; 128 tokens leaves room for a
        # chattier model to show itself without truncating a real sentence.
        # do_sample is not a generation option for the hf provider - pass
        # -M do_sample=false for greedy decoding.
        config=GenerateConfig(max_tokens=128),
        # 1: prompt built from prepare_data_XSum.py by prompt_variant, instead
        # of read from a pre-rendered `prompt` column.
        # 2: format_checks scorer (word range, one sentence, preamble,
        # repetition); pred_words/pred_sentences moved into it from rouge, and
        # sentences no longer counted at every full stop.
        # 3: document_words option, and the default split rebuilt from the Hub
        # when missing. Scoring and default behaviour unchanged.
        version=3,
        metadata={"prompt_variant": prompt_variant, "document_words": document_words},
    )


if __name__ == "__main__":
    # Build the default test split once, on a machine with network access.
    rows = build_default_split()
    print(f"saved {len(rows)} XSum test rows to {DEFAULT_DATASET}")
