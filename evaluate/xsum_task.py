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
from semantic_metrics import score_one, unsupported_entities

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
# The metrics themselves live in semantic_metrics.py, shared with
# evaluate_summarisation.py. See that module for why ROUGE alone is a poor proxy
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
        # nothing to ground, so it is excluded from the aggregate. Inspect's
        # mean() needs a number, so report 0.0 alongside a gradeable flag: the
        # honest figure is sum(entity_support)/sum(entity_gradeable), and plain
        # mean(entity_support) is a lower bound.
        value["entity_gradeable"] = 0.0 if value["entity_support"] is None else 1.0
        if value["entity_support"] is None:
            value["entity_support"] = 0.0
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
