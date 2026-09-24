"""XSum summarisation as an Inspect AI task.

The Inspect equivalent of evaluate_summarisation.py. Same dataset, same
prompts, same ROUGE, but expressed as a Task so it gets Inspect's log format,
`inspect view`, sample-level introspection and its own CLI:

    inspect eval xsum_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false --max-tokens 64

Adapters need the `hf-peft` provider from hf_peft_provider.py, which is
imported below purely for its registration side effect:

    inspect eval xsum_task.py --model hf-peft/google/gemma-3-4b-it \
      -M adapter_path=../models/gemma3-xsum-full-lora -M do_sample=false

`run_inspect_xsum.py` is the wrapper the pipeline uses - it resolves
azureml: references and writes the same results JSON the rest of the repo's
analysis tools read.

PROMPT PARITY: samples take their input from the dataset's `prompt` column,
which prepare_xsum_data.py rendered at data-prep time. Inspect's HF provider
applies the tokenizer's chat template itself, so the model sees exactly what
evaluate_summarisation.py sends it, and neither script re-derives the wording.
"""

import os
import re
import statistics

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from rouge_score import rouge_scorer as rouge_scoring

# Registers the `hf-peft` provider. Imported for the side effect; without it a
# `hf-peft/...` model reference is unknown to Inspect.
from hf_peft_provider import hf_peft  # noqa: F401

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

DEFAULT_DATASET = "../../datasets/xsum_dataset/test"


def count_sentences(text: str) -> int:
    """Rough sentence count - enough to tell one sentence from a paragraph."""
    return sum(text.count(mark) for mark in ".!?") or (1 if text else 0)


def xsum_samples(dataset_path, limit=None, prompt_column="prompt",
                 reference_column="summary"):
    dataset = load_from_disk(dataset_path)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        Sample(
            input=row[prompt_column],
            target=row[reference_column],
            id=row.get("id", index),
            # The source article, for entity_support. A reference-only scorer
            # cannot tell an invented name from a correct one - both are simply
            # absent from a 21-word reference - so grounding needs the document.
            metadata={"document": row.get("document", "")},
        )
        for index, row in enumerate(dataset)
    ]


@scorer(
    metrics={
        "rouge1": [mean(), stderr()],
        "rouge2": [mean(), stderr()],
        "rougeL": [mean(), stderr()],
        # Not quality measures - drift detectors. A merged model sliding back
        # towards base behaviour, or one task's output format leaking into
        # another, moves these before it moves ROUGE.
        "pred_words": [mean()],
        "pred_sentences": [mean()],
    }
)
def rouge():
    """Per-sample ROUGE F1 against the reference summary, plus length stats."""
    scoring = rouge_scoring.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        scores = scoring.score(reference, prediction)
        value = {rouge_type: scores[rouge_type].fmeasure for rouge_type in ROUGE_TYPES}
        value["pred_words"] = float(len(prediction.split()))
        value["pred_sentences"] = float(count_sentences(prediction))
        return Score(
            value=value,
            answer=prediction,
            # Surfaced in `inspect view`, so a sample that looks wrong can be
            # read against its reference without leaving the viewer.
            explanation=f"reference: {reference}",
        )

    return score


# --- semantic scoring -------------------------------------------------------
#
# ROUGE is lexical overlap, and on XSum that is a poor proxy twice over.
#
# It misses correct paraphrase. A 21-word reference and a faithful summary that
# chooses different words score near zero on ROUGE-2 while meaning the same
# thing, so a model rewarded by ROUGE is partly being rewarded for word choice.
# Embedding cosine measures the thing we actually care about.
#
# And it is far too kind to hallucination, which is XSum's characteristic
# failure. Our own measured example: the reference says "Mick Lally ... aged 64",
# the merge produced "Sean Lally ... at the age of 73" and the distilled student
# "Liam Lally ... aged 69". Fluent, on-topic, wrong about the person and the
# number. ROUGE scores those around 0.4; embedding similarity scores them HIGHER
# still, because they are near-identical sentences. Neither notices that the name
# was invented.
#
# entity_support is the metric that does: of the capitalised names and numbers in
# the summary, what fraction appear in the source article? It is a crude proxy
# for faithfulness - string matching, not entity linking, so a correct
# abbreviation or inflection counts as unsupported - which makes it useful for
# COMPARING models on a fixed test set and not as an absolute score.

_EMBEDDER = None


def _embedder():
    """Load the sentence encoder once, or return None if it is unavailable.

    Deliberately non-fatal. Compute nodes have no outbound network, so a missing
    cache must degrade to "no semantic score" rather than fail a run that has
    already spent an hour generating.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer

            _EMBEDDER = SentenceTransformer(
                os.environ.get("XSUM_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
            )
        except Exception as exc:  # noqa: BLE001 - any failure means "skip it"
            print(f"semantic scorer: embedder unavailable ({type(exc).__name__}), "
                  f"reporting similarity as 0.0")
            _EMBEDDER = False
    return _EMBEDDER or None


# Sentence-initial words are capitalised by grammar, not by being names, so the
# first token of each sentence is excluded. Numbers count regardless of position:
# "64" against "73" is the error this is here to catch.
_ENTITY = re.compile(r"\b([A-Z][a-zA-Z'’-]+|\d[\d,.]*)\b")


def entities(text: str) -> list[str]:
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        tokens = _ENTITY.findall(sentence)
        if tokens and sentence.startswith(tokens[0]):
            tokens = tokens[1:]  # grammatical capitalisation, not a name
        found.extend(tokens)
    return found


@scorer(
    metrics={
        "semantic": [mean(), stderr()],
        "entity_support": [mean(), stderr()],
        "entities_unsupported": [mean()],
    }
)
def semantic():
    """Embedding similarity to the reference, and entity grounding in the source."""

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        document = (state.metadata or {}).get("document", "")

        similarity = 0.0
        model = _embedder()
        if model is not None and prediction:
            vectors = model.encode([prediction, reference], normalize_embeddings=True)
            similarity = float(vectors[0] @ vectors[1])

        found = entities(prediction)
        # Case-insensitive substring against the article: the summary may
        # re-inflect a name, and demanding an exact token match would score
        # correct summaries as hallucinated.
        haystack = document.lower()
        unsupported = [e for e in found if e.lower() not in haystack]
        support = 1.0 if not found else (len(found) - len(unsupported)) / len(found)

        return Score(
            value={
                "semantic": similarity,
                "entity_support": support,
                "entities_unsupported": float(len(unsupported)),
            },
            answer=prediction,
            explanation=(
                f"reference: {reference}"
                + (f" | unsupported: {', '.join(unsupported)}" if unsupported else "")
            ),
        )

    return score


@task
def xsum(
    dataset_path: str = DEFAULT_DATASET,
    limit: int | None = None,
    prompt_column: str = "prompt",
    reference_column: str = "summary",
) -> Task:
    """XSum single-sentence summarisation, scored by ROUGE."""
    # Also settable from the CLI as -T dataset_path=..., but an env var lets the
    # pipeline point at a dataset without rewriting the task invocation.
    dataset_path = os.environ.get("XSUM_DATASET", dataset_path)
    if limit is None and os.environ.get("XSUM_LIMIT"):
        limit = int(os.environ["XSUM_LIMIT"])

    return Task(
        dataset=MemoryDataset(
            xsum_samples(dataset_path, limit, prompt_column, reference_column),
            name="xsum",
        ),
        solver=generate(),
        scorer=[rouge(), semantic()],
    )
