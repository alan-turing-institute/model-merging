"""BillSum summarisation as an Inspect AI task.

The summarisation counterpart to xsum_task.py, on a dataset chosen because XSum
turned out to be saturated: every XSum arm - base, experts, merges, distilled
students - landed inside a band narrower than the seed-to-seed spread, so no
merging question could be answered on it. BillSum's inputs are US Congressional
bills (median 1,217 words) and its references are multi-sentence summaries
(median 169 words), which leaves far more room between an untrained model and a
trained one for a merge to fall into.

    inspect eval billsum_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=short

    inspect eval billsum_task.py --model hf-peft/google/gemma-3-4b-it \
      -M adapter_path=../models/gemma3-billsum-full-lora \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=short

evaluate.sbatch builds these commands.

PROMPT PARITY: the prompt is built here, from the bill, by the same
`convert_to_prompt` that train/prepare_billsum_data.py uses to build the training
targets - imported, not copied, so training and evaluation wording cannot drift
apart. `prompt_variant` picks which one, and it MUST match the one the model was
trained on.

TWO TEST SETS. `-T dataset_path=../datasets/billsum/ca_test` scores the same
model on California state bills instead of federal ones, whose references are
roughly twice as long. That is a distribution shift rather than a second sample
of the same thing, and it is the more honest place to look for a merge's
advantage: a method that only ever recovers in-distribution behaviour has
nothing to show there.
"""

import importlib.util
import math
import re
from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from rouge_score import rouge_scorer as rouge_scoring

from semantic_metrics import score_one, unsupported_entities

# rougeLsum, not just rougeL, because BillSum references are multi-sentence.
# rougeL takes the longest common subsequence over the whole string, which
# punishes a summary that covers the same provisions in a different order;
# rougeLsum takes it per sentence and unions, which is what the summarisation
# literature reports on multi-sentence tasks and what BillSum papers quote.
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "billsum" / "test"
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_billsum_data.py"

PROMPT_FIELDS = {
    "short": "messages_short",
    "long": "messages_long",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def load_convert_to_prompt():
    """convert_to_prompt from train/prepare_billsum_data.py, loaded by path
    because train/ is a directory of scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location("prepare_billsum_data", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_to_prompt


def as_sentences(text: str) -> str:
    """One sentence per line, which is how rouge_score's rougeLsum expects its
    input - it splits on newlines and will otherwise treat the whole summary as
    a single sentence, silently making rougeLsum identical to rougeL."""
    return "\n".join(s for s in _SENTENCE_END.split(text.strip()) if s)


def count_sentences(text: str) -> int:
    return sum(text.count(mark) for mark in ".!?") or (1 if text else 0)


def to_chat(messages):
    roles = {"system": ChatMessageSystem, "user": ChatMessageUser}
    return [roles[m["role"]](content=m["content"]) for m in messages]


def billsum_samples(dataset_path, prompt_variant, limit=None,
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
            # The bill text, for entity_support: a reference-only scorer cannot
            # tell an invented agency or dollar figure from a correct one.
            metadata={"document": row["document"]},
        )
        for index, row in enumerate(dataset)
    ]


@scorer(
    metrics={
        "rouge1": [mean(), stderr()],
        "rouge2": [mean(), stderr()],
        "rougeL": [mean(), stderr()],
        "rougeLsum": [mean(), stderr()],
        # Not quality measures - drift and truncation detectors.
        "pred_words": [mean()],
        "ref_words": [mean()],
        "pred_sentences": [mean()],
        # THE GENERATION CAP IS A REAL FAILURE MODE HERE, not a hypothetical: on
        # XSum a 64-token cap truncated 293 of 300 generations on one arm, and
        # the arm's score was read as a model result for weeks. BillSum
        # references are eight times longer, so the cap is correspondingly
        # closer. This counts generations that stopped without terminal
        # punctuation - the signature of hitting max_tokens mid-sentence. If its
        # mean is not near zero, raise max_tokens before reading anything else
        # in the log.
        "unterminated": [mean()],
    }
)
def rouge():
    """Per-sample ROUGE F1 against the reference summary, plus length stats."""
    scoring = rouge_scoring.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        scores = scoring.score(as_sentences(reference), as_sentences(prediction))
        value = {rouge_type: scores[rouge_type].fmeasure for rouge_type in ROUGE_TYPES}
        value["pred_words"] = float(len(prediction.split()))
        value["ref_words"] = float(len(reference.split()))
        value["pred_sentences"] = float(count_sentences(prediction))
        value["unterminated"] = 0.0 if prediction.endswith((".", "!", "?", '"')) else 1.0
        return Score(
            value=value,
            answer=prediction,
            explanation=f"reference: {reference}",
        )

    return score


@scorer(
    metrics={
        "semantic": [mean(), stderr()],
        "entity_support": [mean(), stderr()],
        "entities_unsupported": [mean()],
        "entity_gradeable": [mean()],
    }
)
def semantic():
    """Embedding similarity to the reference, and entity grounding in the bill."""

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        document = (state.metadata or {}).get("document", "")
        value = score_one(prediction, reference, document)
        bad = unsupported_entities(prediction, document)
        # NaN is Inspect's "unscored" marker for a key of a dict-valued score, so
        # a summary that names nobody is left out of entity_support's mean rather
        # than counted as a zero. entity_gradeable keeps the denominator visible.
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
def billsum(
    prompt_variant: str = "short",
    dataset_path: str = str(DEFAULT_DATASET),
    limit: int | None = None,
    reference_column: str = "summary",
    max_tokens: int = 640,
) -> Task:
    """BillSum legislative summarisation, scored by ROUGE and grounding.

    prompt_variant: which prepare_billsum_data.py prompt to evaluate with,
    "short" or "long". Must match what the model was trained on.
    """
    if prompt_variant not in PROMPT_FIELDS:
        raise ValueError(
            f"prompt_variant must be one of {sorted(PROMPT_FIELDS)}, "
            f"got {prompt_variant!r}"
        )

    return Task(
        dataset=MemoryDataset(
            billsum_samples(dataset_path, prompt_variant, limit, reference_column),
            name="billsum",
        ),
        solver=generate(),
        scorer=[rouge(), semantic()],
        # 640, against a reference whose 90th percentile is 340 words (~460
        # tokens). Deliberately generous: see the `unterminated` metric above for
        # what a too-small cap cost us on XSum. ca_test references are longer
        # still - pass -T max_tokens=896 when scoring on that split.
        config=GenerateConfig(max_tokens=max_tokens),
        version=1,
        metadata={"prompt_variant": prompt_variant, "dataset_path": dataset_path},
    )
