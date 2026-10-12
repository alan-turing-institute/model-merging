"""CNN/DailyMail highlight generation as an Inspect AI task, for the domain-split screen.

Headline rougeLsum: highlights are several short sentences, one per line, and
rougeLsum scores them per sentence (see billsum_task.py for why not rougeL). `unterminated` flags generations cut off mid-sentence - if it is not
near zero, raise max_tokens before reading anything else.
"""

import importlib.util
import re
from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from rouge_score import rouge_scorer as rouge_scoring

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "cnndm" / "test"
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_cnndm_data.py"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def load_convert_to_prompt():
    spec = importlib.util.spec_from_file_location("prepare_cnndm_data", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_to_prompt


def as_sentences(text):
    return "\n".join(s for s in _SENTENCE_END.split(text.strip()) if s)


@scorer(metrics={"rougeLsum": [mean(), stderr()], "rouge1": [mean()], "rouge2": [mean()],
                 "pred_words": [mean()], "unterminated": [mean()]})
def rouge():
    scoring = rouge_scoring.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        s = scoring.score(as_sentences(target.text), as_sentences(prediction))
        lsum = s["rougeLsum"].fmeasure
        return Score(
            value={"rougeLsum": lsum, "rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure,
                   "pred_words": float(len(prediction.split())),
                   "unterminated": 0.0 if prediction.endswith((".", "!", "?", '"')) else 1.0},
            answer=prediction, explanation=f"reference: {target.text}")
    return score


@task
def cnndm(dataset_path: str = str(DEFAULT_DATASET), limit: int | None = None,
              max_tokens: int = 384) -> Task:
    convert = load_convert_to_prompt()
    rows = load_from_disk(dataset_path)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    samples = [Sample(input=[ChatMessageUser(content=m["content"]) for m in convert(r)["messages"]],
                      target=r["summary"], id=i, metadata={"article_id": r["id"]})
               for i, r in enumerate(rows)]
    return Task(dataset=MemoryDataset(samples, name="cnndm"), solver=generate(), scorer=rouge(),
                # Highlights run ~55 words; the prep script prints the longest.
                # The untrained models are the arms most likely to hit the cap.
                config=GenerateConfig(max_tokens=max_tokens), version=1,
                metadata={"dataset_path": dataset_path})
