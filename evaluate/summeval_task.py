"""Replays one SummEval system's summaries as an Inspect eval, so summ_metrics.py
can score them exactly as it scores our models (BERTScore, the SummEval toolkit
metrics, the LLM judge).

    uv run inspect eval summeval_task.py -T system=M22 --model mockllm/model

Nothing is generated: the solver sets the output to the system's summary. The
document, the original reference and the expert means are in each sample's
metadata; prepare_summeval.py builds the data. system=reference replays the
original CNN/DailyMail highlight.
"""

from pathlib import Path

from datasets import load_from_disk
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState, solver

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "summeval" / "pairs"
DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")


@solver
def replay():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(model="replay", content=state.metadata["summary"])
        state.completed = True
        return state

    return solve


@task
def summeval(system: str = "M22", dataset_path: str = str(DEFAULT_DATASET)) -> Task:
    rows = load_from_disk(dataset_path).filter(lambda r: r["system"] == system)
    if len(rows) == 0:
        raise ValueError(f"no rows for system {system!r} in {dataset_path}")
    samples = [Sample(input=r["document"], target=r["reference"], id=r["id"],
                      metadata={"document": r["document"], "summary": r["summary"],
                                **{d: r[d] for d in DIMENSIONS}})
               for r in rows]
    return Task(dataset=MemoryDataset(samples, name="summeval"), solver=replay(), version=1,
                metadata={"dataset_path": dataset_path, "system": system})
