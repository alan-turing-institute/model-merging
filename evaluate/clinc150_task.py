"""CLINC150 intent classification as an Inspect AI task, for the domain-split screen.

    inspect eval clinc150_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=32 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=labels

Scored on the 4,500 in-scope test queries (out-of-scope is dropped; see
train/prepare_clinc150_data.py). The test set is balanced - 30 per intent - so
accuracy equals macro recall.

PER-SHARD ACCURACY is what makes the domain split readable: an expert trained on
shard a should be near the full model on shard a and near zero on shard b if the
`short` prompt gives it no way to name labels it never saw. acc_a and acc_b are
NaN outside their shard, which Inspect leaves out of the mean.

`valid` is the share of outputs that are one of the 150 labels at all. A low
value on a merged model is a finding in itself: interference that produces
non-labels rather than wrong labels.
"""

import importlib.util
import json
import math
import re
from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "clinc150" / "test"
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_clinc150_data.py"
DOMAINS_FILE = REPO_ROOT / "train" / "clinc150_domains.json"
PROMPT_FIELDS = {"short": "messages_short", "labels": "messages_labels"}
LABELS = frozenset(i for v in json.loads(DOMAINS_FILE.read_text()).values() for i in v)


def load_convert_to_prompt():
    spec = importlib.util.spec_from_file_location("prepare_clinc150_data", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_to_prompt


def normalise(text: str) -> str:
    """First line, lowercased, quotes and trailing punctuation stripped, spaces
    and hyphens to underscores - so "Freeze account." scores as freeze_account,
    but a sentence explaining the label does not."""
    first = (text.strip().splitlines() or [""])[0]
    first = first.strip().strip("`'\"*").strip().rstrip(".!?:;,").lower()
    return re.sub(r"[\s\-]+", "_", first)


@scorer(metrics={"accuracy": [mean(), stderr()], "acc_a": [mean()], "acc_b": [mean()],
                 "valid": [mean()]})
def intent_match():
    async def score(state: TaskState, target: Target) -> Score:
        prediction = normalise(state.output.completion)
        correct = float(prediction == target.text)
        shard = (state.metadata or {}).get("shard")
        return Score(
            value={"accuracy": correct,
                   "acc_a": correct if shard == "a" else math.nan,
                   "acc_b": correct if shard == "b" else math.nan,
                   "valid": float(prediction in LABELS)},
            answer=prediction,
            explanation=f"label: {target.text} ({(state.metadata or {}).get('domain')})",
        )
    return score


@task
def clinc150(prompt_variant: str = "short", dataset_path: str = str(DEFAULT_DATASET),
             limit: int | None = None, max_tokens: int = 16) -> Task:
    """prompt_variant must match training; the base model is scored on both."""
    if prompt_variant not in PROMPT_FIELDS:
        raise ValueError(f"prompt_variant must be one of {sorted(PROMPT_FIELDS)}")
    convert = load_convert_to_prompt()
    rows = load_from_disk(dataset_path)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    samples = [
        Sample(input=[ChatMessageUser(content=m["content"])
                      for m in convert(r)[PROMPT_FIELDS[prompt_variant]]],
               target=r["label"], id=i,
               metadata={"domain": r["domain"], "shard": r["shard"]})
        for i, r in enumerate(rows)
    ]
    return Task(
        dataset=MemoryDataset(samples, name="clinc150"),
        solver=generate(),
        scorer=intent_match(),
        # The longest label is 9 Gemma tokens; 16 leaves room for a stray quote
        # without letting an untrained model ramble at length.
        config=GenerateConfig(max_tokens=max_tokens),
        version=1,
        metadata={"prompt_variant": prompt_variant, "dataset_path": dataset_path},
    )
