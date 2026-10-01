"""Reddit crime classification as an Inspect AI task.

The Inspect equivalent of old/evaluate.py, and the evaluator the KD and cross-task
arms use for this dataset. Same prompt, same label parsing, same metric
definitions - what changes is that it runs through Inspect, so it gets a
browsable log, per-sample introspection, and the provider's batching rather
than old/evaluate.py's one-example-at-a-time loop.

PROMPT PARITY: the user turn is built here, from the post, by the same
`convert` that train/prepare_data.py uses to build the training data -
imported, not copied, so training and evaluation wording cannot drift apart.

The negation handling in convert_to_label ("not really about crime") is the
difference between a plausible-looking accuracy and a correct one.

    inspect eval crime_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16

A LoRA adapter on a base model uses the hf-peft provider (hf_peft_provider.py,
registered as an Inspect plugin by this project's pyproject.toml):

    inspect eval crime_task.py --model hf-peft/google/gemma-3-4b-it \
      -M adapter_path=../models/gemma3-crime-full-lora \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16
"""

import importlib.util
import re
from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Target,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

# Resolved from this file rather than the working directory, so the defaults
# hold wherever `inspect eval` is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "crime_dataset" / "test"
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_data.py"
LABELS = {0: "not_crime", 1: "crime"}


def load_convert():
    """convert from train/prepare_data.py.

    Loaded by path because train/ is a directory of scripts, not an importable
    package.
    """
    spec = importlib.util.spec_from_file_location("prepare_data", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


NEGATION_RE = re.compile(r"\b(not|no|never)\b|n't")


def convert_to_label(text):
    text = text.lower()
    if "not_crime" in text:
        return 0
    crime_index = text.find("crime")
    if crime_index == -1:
        return None
    # A negation word anywhere before "crime" (e.g. "not a crime", "not
    # really about crime") means the answer is negative, not positive.
    if NEGATION_RE.search(text[:crime_index]):
        return 0
    return 1


def crime_samples(dataset_path, limit=None, text_column="posts_name",
                  label_column="label"):
    convert = load_convert()
    dataset = load_from_disk(str(dataset_path))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        Sample(
            # [0]["content"]: convert returns the user/assistant pair, and
            # Inspect's solver wraps the input into a user turn itself.
            input=convert({"posts_name": row[text_column],
                           "label": row[label_column]})["messages"][0]["content"],
            target=LABELS[int(row[label_column])],
            id=index,
            metadata={"label": int(row[label_column])},
        )
        for index, row in enumerate(dataset)
    ]


def scored_prediction(sample_score: SampleScore) -> tuple[int, int]:
    """(predicted, actual), with an unparsed generation as the wrong label.

    Matches old/evaluate.py: an unparsed generation is scored wrong rather than
    dropped, so the denominator of every metric is the whole test set.
    """
    meta = sample_score.score.metadata
    actual = meta["actual"]
    predicted = meta["predicted"]
    return (1 - actual if predicted is None else predicted), actual


@metric
def confusion() -> Metric:
    """Confusion-matrix counts, with crime as the positive class."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        pairs = [scored_prediction(s) for s in scores]
        return {
            "tp": float(sum(p == 1 and a == 1 for p, a in pairs)),
            "fp": float(sum(p == 1 and a == 0 for p, a in pairs)),
            "fn": float(sum(p == 0 and a == 1 for p, a in pairs)),
            "tn": float(sum(p == 0 and a == 0 for p, a in pairs)),
        }

    return compute


@metric
def precision() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        c = confusion()(scores)
        predicted_positive = c["tp"] + c["fp"]
        return c["tp"] / predicted_positive if predicted_positive else 0.0

    return compute


@metric
def recall() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        c = confusion()(scores)
        actual_positive = c["tp"] + c["fn"]
        return c["tp"] / actual_positive if actual_positive else 0.0

    return compute


@metric
def f1() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        c = confusion()(scores)
        denominator = 2 * c["tp"] + c["fp"] + c["fn"]
        return 2 * c["tp"] / denominator if denominator else 0.0

    return compute


# precision/recall/f1/confusion read the per-sample labels from metadata, so
# which value key they hang off is arbitrary; accuracy is the natural home.
@scorer(
    metrics={
        "accuracy": [mean(), stderr(), precision(), recall(), f1(), confusion()],
        "unparsed": [mean()],
    }
)
def crime_label():
    """Exact-label scoring, with an unparsed generation counted as wrong."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion.strip()
        predicted = convert_to_label(completion)
        actual = 1 if target.text == "crime" else 0

        # An unparsed generation is wrong, never dropped (see scored_prediction).
        correct = predicted is not None and predicted == actual

        return Score(
            value={"accuracy": float(correct), "unparsed": float(predicted is None)},
            answer=completion,
            # Read by precision/recall/f1/confusion, and kept per sample so two
            # runs can be paired (e.g. McNemar) from the log alone.
            metadata={"predicted": predicted, "actual": actual},
        )

    return score


@task
def crime(
    dataset_path: str = str(DEFAULT_DATASET),
    limit: int | None = None,
    text_column: str = "posts_name",
    label_column: str = "label",
) -> Task:
    """Binary crime/not_crime classification of Reddit posts."""
    return Task(
        dataset=MemoryDataset(
            crime_samples(dataset_path, limit, text_column, label_column),
            name="crime",
        ),
        solver=generate(),
        scorer=crime_label(),
        # A one-word answer; 16 tokens leaves room to see a model that has
        # started explaining itself. do_sample is not a generation option for
        # the hf provider - pass -M do_sample=false for greedy decoding.
        config=GenerateConfig(max_tokens=16),
        # 1: prompt from train/prepare_data.py's convert; precision, recall,
        # F1 and confusion counts computed in the task.
        version=1,
    )
