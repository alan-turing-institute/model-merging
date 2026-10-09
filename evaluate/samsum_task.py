"""SAMSum dialogue summarisation as an Inspect AI task.

The third summarisation task in the programme, after XSum (saturated: every arm
landed inside a band narrower than the seed-to-seed spread) and BillSum (headroom
at 500->1,000 rows, flat by 4,000). SAMSum is a different bet from BillSum:
BillSum widened the base-to-trained gap by making the TARGET long and structured,
SAMSum keeps the target short - 20.3 words, against XSum's 21.0 - and moves the
INPUT out of distribution instead, to messenger dialogue with a summary
convention (third person, named participants, reported intent) an
instruction-tuned model does not produce unprompted.

    inspect eval samsum_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=short

    inspect eval samsum_task.py --model hf-peft/google/gemma-3-4b-it \
      -M adapter_path=../models/gemma3-samsum-n8000-lora \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=short

evaluate.sbatch builds these commands.

PROMPT PARITY: the prompt is built here, from the dialogue, by the same
`convert_to_prompt` that train/prepare_samsum_data.py uses to build the training
targets - imported, not copied, so training and evaluation wording cannot drift
apart. `prompt_variant` picks which one, and it MUST match the one the model was
trained on.

WHY entity_support MATTERS MORE HERE than on the other two tasks. SAMSum
dialogues name their participants, and the characteristic failure of a dialogue
summariser is not inventing facts but attributing a real action to the wrong
speaker. That is an entity error, and it is invisible to ROUGE, which scores a
summary containing the right names in the wrong roles almost as highly as the
correct one.
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

# All four, as on BillSum, but note which one to read. SAMSum references are one
# or two sentences, so rougeLsum and rougeL are nearly identical here and neither
# carries the extra information it does on BillSum. The SAMSum literature quotes
# rouge1/rouge2/rougeL, and rouge1 is the headline for the scaling curve.
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "samsum" / "test"
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_samsum_data.py"

PROMPT_FIELDS = {
    "short": "messages_short",
    "long": "messages_long",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def load_convert_to_prompt():
    """convert_to_prompt from train/prepare_samsum_data.py, loaded by path
    because train/ is a directory of scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location("prepare_samsum_data", PREPARE_SCRIPT)
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


def samsum_samples(dataset_path, prompt_variant, limit=None,
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
            # The dialogue, for entity_support. Carried per sample rather than
            # paired by list position - the defect that made the old XSum
            # re-scoring scripts unsafe to reorder.
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
        # On XSum a 64-token cap truncated 293 of 300 generations on one arm and
        # the arm's score was read as a model result for weeks. If this mean is
        # not near zero, raise max_tokens before reading anything else.
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
    """Embedding similarity to the reference, and entity grounding in the dialogue."""

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        document = (state.metadata or {}).get("document", "")
        value = score_one(prediction, reference, document)
        bad = unsupported_entities(prediction, document)
        # NaN is Inspect's "unscored" marker for a key of a dict-valued score, so
        # a summary that names nobody is left out of entity_support's mean rather
        # than counted as a zero. entity_gradeable keeps the denominator visible
        # - and on SAMSum it should be near 1.0, because the references name
        # their participants. A low value means the model stopped using names,
        # which is itself a finding.
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
def samsum(
    prompt_variant: str = "short",
    dataset_path: str = str(DEFAULT_DATASET),
    limit: int | None = None,
    reference_column: str = "summary",
    max_tokens: int = 128,
) -> Task:
    """SAMSum dialogue summarisation, scored by ROUGE and grounding.

    prompt_variant: which prepare_samsum_data.py prompt to evaluate with,
    "short" or "long". Must match what the model was trained on.
    """
    if prompt_variant not in PROMPT_FIELDS:
        raise ValueError(
            f"prompt_variant must be one of {sorted(PROMPT_FIELDS)}, "
            f"got {prompt_variant!r}"
        )

    return Task(
        dataset=MemoryDataset(
            samsum_samples(dataset_path, prompt_variant, limit, reference_column),
            name="samsum",
        ),
        solver=generate(),
        scorer=[rouge(), semantic()],
        # 128, against a reference whose observed maximum is 64 words (~90
        # tokens). Generous on purpose, and the `unterminated` metric is the
        # check that it was generous enough. It also matters that the UNTRAINED
        # base model is scored at this cap: it has not learned to stop, so it
        # will run long, and clipping it at the reference length would flatter
        # the trained arms by construction - the n=0 point would be measuring
        # the cap rather than the model.
        config=GenerateConfig(max_tokens=max_tokens),
        version=1,
        metadata={"prompt_variant": prompt_variant, "dataset_path": dataset_path},
    )
