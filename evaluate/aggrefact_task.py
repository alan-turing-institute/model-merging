"""Replays LLM-AggreFact's claims as Inspect evals, so summ_metrics.py can run
the LLM judge and DeepEval on them like any summary (see prepare_aggrefact.py).

    uv run inspect eval aggrefact_task.py@aggrefact_cnn -T split=test --model mockllm/model

Nothing is generated: the output is the benchmark's claim, the document is in
metadata, and so is the human label for aggrefact_eval.py.
"""

from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState, solver

DATA = Path(__file__).resolve().parent.parent / "datasets" / "aggrefact"


@solver
def replay():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(model="replay", content=state.metadata["claim"])
        state.completed = True
        return state

    return solve


def _task(subset, split):
    rows = load_from_disk(str(DATA / f"{subset}_{split}"))
    samples = [Sample(input=r["doc"], target="", id=r["id"],
                      metadata={"document": r["doc"], "claim": r["claim"], "label": r["label"]}) for r in rows]
    return Task(dataset=MemoryDataset(samples, name=f"aggrefact_{subset}"), solver=replay(), version=1,
                metadata={"split": split})


@task
def aggrefact_cnn(split: str = "test") -> Task:
    return _task("cnn", split)


@task
def aggrefact_xsum(split: str = "test") -> Task:
    return _task("xsum", split)
