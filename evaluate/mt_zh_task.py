"""English->Chinese UN/news translation as an Inspect AI task, for the
domain-split screen.

    inspect eval mt_zh_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=16 -M do_sample=false -M dtype=bfloat16

HEADLINE: sentence-level chrF (sacrebleu, character n-grams only, word_order=0),
averaged. Not chrF++: its word n-grams split on whitespace, which Chinese does
not use, so they would add noise rather than signal. Mean sentence
chrF++ is not corpus chrF++, but it is what a paired bootstrap over sentences
needs, and the screen's comparisons are all within this one harness. COMET is
scored separately in metrics/, whose environment it needs.

chrf_un and chrf_news are NaN outside their domain. `truncated` is the share
of generations stopped by max_tokens rather than by the model - a translation cut
off mid-sentence scores low for a reason that has nothing to do with the model.
If it is not near zero, raise max_tokens before reading anything else.
"""

import importlib.util
import math
from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from sacrebleu.metrics import CHRF

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "mt_zh" / "test"
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_mt_zh_data.py"


def load_convert_to_prompt():
    spec = importlib.util.spec_from_file_location("prepare_mt_zh_data", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_to_prompt


@scorer(metrics={"chrf": [mean(), stderr()], "chrf_un": [mean()], "chrf_news": [mean()],
                 "truncated": [mean()], "length_ratio": [mean()]})
def chrf_pp():
    metric = CHRF(word_order=0)

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        value = metric.sentence_score(prediction, [target.text]).score
        domain = (state.metadata or {}).get("domain")
        return Score(
            value={"chrf": value,
                   "chrf_un": value if domain == "un" else math.nan,
                   "chrf_news": value if domain == "news" else math.nan,
                   "truncated": float(state.output.stop_reason == "max_tokens"),
                   "length_ratio": len(prediction) / max(1, len(target.text))},
            answer=prediction,
            explanation=f"reference: {target.text}",
        )
    return score


@task
def mt_zh(dataset_path: str = str(DEFAULT_DATASET), limit: int | None = None,
              max_tokens: int = 512) -> Task:
    convert = load_convert_to_prompt()
    rows = load_from_disk(dataset_path)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    samples = [
        Sample(input=[ChatMessageUser(content=m["content"]) for m in convert(r)["messages"]],
               target=r["zh"], id=i, metadata={"domain": r["domain"], "shard": r["shard"]})
        for i, r in enumerate(rows)
    ]
    return Task(
        dataset=MemoryDataset(samples, name="mt_zh"),
        solver=generate(),
        scorer=chrf_pp(),
        # 512 against references whose maximum the prep script prints; legal
        # sentences run long. The `truncated` metric is the check.
        config=GenerateConfig(max_tokens=max_tokens),
        version=1,
        metadata={"dataset_path": dataset_path},
    )
