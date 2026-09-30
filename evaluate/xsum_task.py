"""XSum summarisation as an Inspect AI task.

The Inspect equivalent of evaluate_summarisation.py - ROUGE plus the semantic
and entity-grounding scores - expressed as a Task so it gets Inspect's log
format, `inspect view`, sample-level introspection and its own CLI:

    inspect eval xsum_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false -T prompt_variant=short

A full model saved on disk (config.json + model.safetensors) loads with the
plain hf provider; the name after hf/ is only a label, model_path is what loads:

    inspect eval xsum_task.py --model hf/gemma3-xsum-self-dist-short \
      -M model_path=../models/gemma3-xsum-self-dist-short \
      -M batch_size=8 -M do_sample=false -T prompt_variant=short

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
"""

import importlib.util
import math
from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from rouge_score import rouge_scorer as rouge_scoring

from semantic_metrics import score_one, unsupported_entities

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

# Resolved from this file rather than the working directory, so the default
# holds wherever `inspect eval` is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "xsum_prompts" / "test"
PREPARE_SCRIPT = REPO_ROOT / "self-distill" / "prepare_data_XSum.py"

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


def count_sentences(text: str) -> int:
    """Rough sentence count - enough to tell one sentence from a paragraph."""
    return sum(text.count(mark) for mark in ".!?") or (1 if text else 0)


def to_chat(messages):
    roles = {"system": ChatMessageSystem, "user": ChatMessageUser}
    return [roles[m["role"]](content=m["content"]) for m in messages]


def xsum_samples(dataset_path, prompt_variant, limit=None,
                 reference_column="summary"):
    field = PROMPT_FIELDS[prompt_variant]
    convert_to_prompt = load_convert_to_prompt()
    dataset = load_from_disk(str(dataset_path))
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
        # nothing to ground. NaN is Inspect's "unscored" marker for a key of a
        # dict-valued score: that sample is left out of entity_support's mean
        # and stderr, so the logged figure is the mean over gradeable summaries
        # only, as evaluate_summarisation.py computes it. entity_gradeable
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
) -> Task:
    """XSum single-sentence summarisation, scored by ROUGE and grounding.

    prompt_variant: which prepare_data_XSum.py prompt to evaluate with - one of
    "short", "long" or "teacher". Must match what the model was trained on.
    """
    if prompt_variant not in PROMPT_FIELDS:
        raise ValueError(
            f"prompt_variant must be one of {sorted(PROMPT_FIELDS)}, "
            f"got {prompt_variant!r}"
        )

    return Task(
        dataset=MemoryDataset(
            xsum_samples(dataset_path, prompt_variant, limit, reference_column),
            name="xsum",
        ),
        solver=generate(),
        scorer=[rouge(), semantic()],
        # The models were trained earlier with a teacher which had 128 tokens.
        # XSum references average ~21 words; 128 tokens leaves room for a
        # chattier model to show itself without truncating a real sentence.
        # do_sample is not a generation option for the hf provider - pass
        # -M do_sample=false for greedy decoding.
        config=GenerateConfig(max_tokens=128),
        # 1: prompt built from prepare_data_XSum.py by prompt_variant, instead
        # of read from a pre-rendered `prompt` column.
        version=1,
        metadata={"prompt_variant": prompt_variant},
    )
