"""Reddit crime classification as an Inspect AI task.

The Inspect equivalent of evaluate.py, and the evaluator the KD and cross-task
arms use for this dataset. Same prompt, same label parsing, same metric
definitions - what changes is that it runs through Inspect, so it gets a
browsable log, per-sample introspection, and the provider's batching rather
than evaluate.py's one-example-at-a-time loop.

PROMPT AND PARSING ARE IMPORTED, NOT REIMPLEMENTED. `build_messages` and
`convert_to_label` come from evaluate.py so the two evaluators cannot drift
apart. That matters more than it sounds: the negation handling in
convert_to_label ("not really about crime") is the difference between a
plausible-looking accuracy and a correct one.

KNOWN PROMPT DRIFT, PRESERVED DELIBERATELY. prepare_data.py trains on
"either 'crime' or 'not_crime'" (single quotes) while evaluate.py asks with
"either \"crime\" or \"not_crime\"" (double quotes). Every existing crime
result was produced with the evaluation wording, so this task uses it too -
changing it would make new numbers incomparable with the pilot. It is recorded
here rather than silently repaired; the fix belongs with a re-run, not with a
change of evaluator.

    inspect eval crime_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false --max-tokens 16

Adapters need run_inspect_crime.py, which registers the hf-peft provider first.
"""

import os

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState, generate

# Registers the `hf-peft` provider (adapter support, which Inspect's built-in
# hf provider does not have).
from hf_peft_provider import hf_peft  # noqa: F401

# Single source of truth for the prompt and the label parsing.
from evaluate import build_messages, convert_to_label

DEFAULT_DATASET = "../../datasets/crime_dataset/test"
LABELS = {0: "not_crime", 1: "crime"}


def crime_samples(dataset_path, limit=None, text_column="posts_name",
                  label_column="label"):
    dataset = load_from_disk(dataset_path)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        Sample(
            # [0]["content"] because build_messages returns a chat turn and
            # Inspect's solver wraps the input into one itself.
            input=build_messages(row[text_column])[0]["content"],
            target=LABELS[int(row[label_column])],
            id=index,
            metadata={"label": int(row[label_column])},
        )
        for index, row in enumerate(dataset)
    ]


@scorer(metrics={"accuracy": [mean()], "unparsed": [mean()]})
def crime_label():
    """Exact-label scoring, with an unparsed generation counted as wrong."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion.strip()
        predicted = convert_to_label(completion)
        actual = 1 if target.text == "crime" else 0

        # Matches evaluate.py: a generation that parses to neither label is
        # scored wrong rather than dropped, so every example counts towards the
        # metrics and the denominator is the whole test set.
        correct = predicted is not None and predicted == actual

        return Score(
            value={"accuracy": float(correct), "unparsed": float(predicted is None)},
            answer=completion,
            # The wrapper rebuilds precision/recall/F1 from these, so they have
            # to survive into the log per sample.
            metadata={"predicted": predicted, "actual": actual},
        )

    return score


@task
def crime(
    dataset_path: str = DEFAULT_DATASET,
    limit: int | None = None,
    text_column: str = "posts_name",
    label_column: str = "label",
) -> Task:
    """Binary crime/not_crime classification of Reddit posts."""
    dataset_path = os.environ.get("CRIME_DATASET", dataset_path)
    if limit is None and os.environ.get("CRIME_LIMIT"):
        limit = int(os.environ["CRIME_LIMIT"])

    return Task(
        dataset=MemoryDataset(
            crime_samples(dataset_path, limit, text_column, label_column),
            name="crime",
        ),
        solver=generate(),
        scorer=crime_label(),
    )
