"""CLINC150 intent classification as an Inspect AI task.

Each in-scope test query (datasets/clinc150/clinc150_labels_dataset/test, or
clinc150_dataset/test, which holds the same 4,500 queries, 30 per intent, so
accuracy equals macro recall) is put to the model
with the instruction to answer with the intent label only, and the answer is
scored by exact match after normalising.

PROMPT. The user turn is built by `convert_to_prompt` from
train/prepare_data_clinc150.py, the function that built the training data. It
is imported, not copied, so training and evaluation wording cannot drift apart.
prompt_variant picks the wording: "short" (the instruction and the query, as in
training) or "labels" (all 150 labels listed as well). It must match the
training prompt; an untrained model is worth scoring on both, since without the
label list it cannot know the label strings.

PARSING (normalise). The answer's first line, lowercased, with quotes and
trailing punctuation stripped and spaces and hyphens turned into underscores:
"Freeze account." scores as freeze_account, a sentence explaining the label
does not.

METRICS: accuracy (with stderr); acc_shard1 and acc_shard2, the accuracy on
the queries of each domain shard (train/prepare_data_clinc150.py's SHARDS), NaN
outside it, which is what makes a domain split readable: an expert should be
near the full model at home and near zero away; and `valid`, the share of
answers that are one of the 150 labels at all, so that a merge which produces
non-labels rather than wrong labels is visible.

USAGE. From evaluate/:
    inspect eval clinc150_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=32 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=labels
or on the cluster,
    sbatch evaluate.sbatch clinc150 google/gemma-3-4b-it \
      --adapter ../models/clinc150/gemma3_4b_clinc150 -T prompt_variant=short

Task arguments (-T): prompt_variant, dataset_path, limit, max_tokens.
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

# Resolved from this file rather than the working directory, so the defaults
# hold wherever `inspect eval` is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
# Only each test row's query, label and domain are read (the prompt is built
# below), and those are the same in the labels and the short datasets, so either
# serves; the labels one, the default of prepare_data_clinc150.py, if it exists.
DEFAULT_DATASET = next(
    (p for p in (REPO_ROOT / "datasets" / "clinc150" / f"{name}_dataset" / "test"
                 for name in ("clinc150_labels", "clinc150")) if p.exists()),
    REPO_ROOT / "datasets" / "clinc150" / "clinc150_labels_dataset" / "test",
)
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_data_clinc150.py"
DOMAINS_FILE = REPO_ROOT / "train" / "clinc150_domains.json"
PROMPT_VARIANTS = ("short", "labels")
# Every intent in train/clinc150_domains.json, for `valid`.
LABELS = frozenset(i for v in json.loads(DOMAINS_FILE.read_text()).values() for i in v)


def load_convert_to_prompt():
    """convert_to_prompt from train/prepare_data_clinc150.py.

    Loaded by path because train/ is a directory of scripts, not an importable
    package.
    """
    spec = importlib.util.spec_from_file_location("prepare_data_clinc150", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_to_prompt


def normalise(text):
    first = (text.strip().splitlines() or [""])[0]
    first = first.strip().strip("`'\"*").strip().rstrip(".!?:;,").lower()
    return re.sub(r"[\s\-]+", "_", first)


@scorer(metrics={"accuracy": [mean(), stderr()], "acc_shard1": [mean()], "acc_shard2": [mean()],
                 "valid": [mean()]})
def intent_match():
    async def score(state: TaskState, target: Target) -> Score:
        prediction = normalise(state.output.completion)
        correct = float(prediction == target.text)
        shard = (state.metadata or {}).get("shard")
        return Score(
            value={"accuracy": correct,
                   "acc_shard1": correct if shard == 1 else math.nan,
                   "acc_shard2": correct if shard == 2 else math.nan,
                   "valid": float(prediction in LABELS)},
            answer=prediction,
            explanation=f"label: {target.text} ({(state.metadata or {}).get('domain')})",
        )
    return score


@task
def clinc150(prompt_variant: str = "short", dataset_path: str = str(DEFAULT_DATASET),
             limit: int | None = None, max_tokens: int = 16) -> Task:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"prompt_variant must be one of {PROMPT_VARIANTS}, got {prompt_variant!r}")
    convert = load_convert_to_prompt()
    rows = load_from_disk(dataset_path)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    samples = [
        Sample(input=[ChatMessageUser(content=m["content"]) for m in convert(r)[prompt_variant]],
               target=r["label"], id=i, metadata={"domain": r["domain"], "shard": r["shard"]})
        for i, r in enumerate(rows)
    ]
    return Task(
        dataset=MemoryDataset(samples, name="clinc150"),
        solver=generate(),
        scorer=intent_match(),
        # The longest label is 9 Gemma tokens; 16 leaves room for a stray quote
        # without letting an untrained model ramble.
        config=GenerateConfig(max_tokens=max_tokens),
        version=1,
        metadata={"prompt_variant": prompt_variant, "dataset_path": dataset_path},
    )
