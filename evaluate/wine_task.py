"""Wine review points prediction as an Inspect AI task.

The model reads a Wine Enthusiast review and answers with the points rating
(80-100). Scored against the known rating by:

- abs_error   - mean absolute error (MAE), the headline number, with rmse
                alongside it, and median_baseline_mae: the MAE of always
                answering the train median (88) on the same samples. Scores
                are bunched (train std ~3.15), so that constant already gets
                MAE ~2.5 - a model is only useful if it clearly beats it.
- pearson, spearman - correlation of predicted with true points. A model that
                has collapsed to answering "88" for everything gets a decent
                MAE but a correlation near zero (NaN if it never varies).
- pred_std, true_std - spread of predictions vs truth; a much smaller pred_std
                is the model hedging towards the middle.
- signed_error - mean of predicted - true (bias), plus MAE and bias per score
                band, since regression to the mean overrates poor wines and
                underrates great ones.
- exact, within_1, within_2 - share of predictions at most 0/1/2 points off.
- out_of_range - share of answers outside the 80-100 scale (e.g. "78").
- unanswered  - share of answers with no number in them at all.

The first number in an answer is the prediction, wherever it falls: "78" is
scored as 78, 2 below an 80 rather than replaced by anything. An answer with
no number is unanswered: it is left out of every other metric (they are over
the answered samples only) and reported separately as `unanswered`. Read MAE
with `unanswered` next to it - two models with different unanswered rates are
being averaged over different samples.

PROMPT PARITY: the user turn is built here by the same `convert` that
train/prepare_data_wine.py uses to build the training data - imported, not
copied, so training and evaluation wording cannot drift apart.

    inspect eval wine_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16

A LoRA adapter on a base model uses the hf-peft provider:

    inspect eval wine_task.py --model hf-peft/google/gemma-3-4b-it \
      -M adapter_path=../models/gemma3-4b-wine-full-lora \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16
"""

import importlib.util
import math
import re
import statistics
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
DEFAULT_DATASET = REPO_ROOT / "datasets" / "wine_dataset" / "test"
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_data_wine.py"

MIN_POINTS, MAX_POINTS = 80, 100
# Median of the training points: the constant-guess baseline.
BASELINE = 88
BANDS = [(80, 84), (85, 89), (90, 94), (95, 100)]


def load_convert():
    """convert from train/prepare_data_wine.py.

    Loaded by path because train/ is a directory of scripts, not an importable
    package.
    """
    spec = importlib.util.spec_from_file_location("prepare_data_wine", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def parse_points(text):
    """The first number in the answer, or None if it has no number.

    Taking the first number means "87/100" or "87 points" parse as 87, while a
    model that has started explaining ("This wine scores 87") still parses. A
    number off the scale is kept as it is - "78" is the model's estimate, and
    out_of_range records how often that happens.
    """
    match = NUMBER_RE.search(text)
    return None if match is None else float(match.group())


def wine_samples(dataset_path, limit=None, text_column="description",
                 points_column="points"):
    convert = load_convert()
    dataset = load_from_disk(str(dataset_path))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        Sample(
            # [0]["content"]: convert returns the user/assistant pair, and
            # Inspect's solver wraps the input into a user turn itself.
            input=convert({"description": row[text_column],
                           "points": row[points_column]})["messages"][0]["content"],
            target=str(row[points_column]),
            id=index,
            metadata={"points": int(row[points_column])},
        )
        for index, row in enumerate(dataset)
    ]


def pairs(scores: list[SampleScore]) -> list[tuple[float, float]]:
    """(predicted, actual) per answered sample; unanswered ones are left out."""
    return [
        (s.score.metadata["predicted"], s.score.metadata["actual"])
        for s in scores
        if s.score.metadata["predicted"] is not None
    ]


@metric
def rmse() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        errors = [(p - a) ** 2 for p, a in pairs(scores)]
        return math.sqrt(sum(errors) / len(errors)) if errors else math.nan

    return compute


@metric
def median_baseline_mae() -> Metric:
    """MAE of always answering 88 on the same (answered) samples - the bar a
    model has to clear."""

    def compute(scores: list[SampleScore]) -> float:
        errors = [abs(a - BASELINE) for _, a in pairs(scores)]
        return sum(errors) / len(errors) if errors else math.nan

    return compute


def correlation(scores: list[SampleScore], method: str) -> float:
    answered = pairs(scores)
    predicted, actual = zip(*answered) if answered else ((), ())
    try:
        return statistics.correlation(predicted, actual, method=method)
    except statistics.StatisticsError:
        # Fewer than two samples, or a constant column - most likely a model
        # that answers the same number every time. Correlation is undefined.
        return math.nan


@metric
def pearson() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return correlation(scores, "linear")

    return compute


@metric
def spearman() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return correlation(scores, "ranked")

    return compute


@metric
def spread() -> Metric:
    """Standard deviation of predicted and of true points."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        answered = pairs(scores)
        predicted, actual = zip(*answered) if answered else ((), ())
        return {
            "pred_std": statistics.pstdev(predicted) if predicted else math.nan,
            "true_std": statistics.pstdev(actual) if actual else math.nan,
        }

    return compute


@metric
def by_band() -> Metric:
    """MAE, bias and sample count within each band of true points."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        result = {}
        for low, high in BANDS:
            band = [(p, a) for p, a in pairs(scores) if low <= a <= high]
            name = f"{low}_{high}"
            result[f"n_{name}"] = float(len(band))
            if band:
                result[f"mae_{name}"] = sum(abs(p - a) for p, a in band) / len(band)
                result[f"bias_{name}"] = sum(p - a for p, a in band) / len(band)
            else:
                result[f"mae_{name}"] = math.nan
                result[f"bias_{name}"] = math.nan
        return result

    return compute


# rmse/correlation/spread/band metrics read the per-sample pair from metadata,
# so which value key they hang off is arbitrary; abs_error is the natural home.
@scorer(
    metrics={
        "abs_error": [mean(), stderr(), rmse(), median_baseline_mae(), pearson(),
                      spearman(), spread(), by_band()],
        "signed_error": [mean()],
        "exact": [mean(), stderr()],
        "within_1": [mean(), stderr()],
        "within_2": [mean(), stderr()],
        "out_of_range": [mean()],
        # Over all samples, unlike every other metric.
        "unanswered": [mean()],
    }
)
def points_error():
    """Distance of the predicted rating from the true one."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion.strip()
        predicted = parse_points(completion)
        actual = int(target.text)

        if predicted is None:
            # NaN is Inspect's "unscored" marker for a key of a dict-valued
            # score: the sample is left out of that key's mean and stderr, so
            # every metric but unanswered is over the answered samples only.
            value = {key: math.nan for key in
                     ("abs_error", "signed_error", "exact", "within_1",
                      "within_2", "out_of_range")}
            value["unanswered"] = 1.0
        else:
            error = predicted - actual
            value = {
                "abs_error": abs(error),
                "signed_error": error,
                # Round so a fractional answer like 87.4 can still be exact.
                "exact": float(round(abs(error)) == 0),
                "within_1": float(round(abs(error)) <= 1),
                "within_2": float(round(abs(error)) <= 2),
                "out_of_range": float(not MIN_POINTS <= predicted <= MAX_POINTS),
                "unanswered": 0.0,
            }

        return Score(
            value=value,
            answer=completion,
            explanation=f"true {actual}, predicted {predicted}",
            # Read by the pair-based metrics, and kept per sample so two runs
            # can be paired (e.g. a Wilcoxon test on abs_error) from the log alone.
            metadata={"predicted": predicted, "actual": actual},
        )

    return score


@task
def wine(
    dataset_path: str = str(DEFAULT_DATASET),
    limit: int | None = None,
    text_column: str = "description",
    points_column: str = "points",
) -> Task:
    """Predict a wine's Wine Enthusiast points (80-100) from its review."""
    return Task(
        dataset=MemoryDataset(
            wine_samples(dataset_path, limit, text_column, points_column),
            name="wine",
        ),
        solver=generate(),
        scorer=points_error(),
        # A two-digit answer; 16 tokens leaves room to see a model that has
        # started explaining itself. do_sample is not a generation option for
        # the hf provider - pass -M do_sample=false for greedy decoding.
        config=GenerateConfig(max_tokens=16),
        version=1,
    )
