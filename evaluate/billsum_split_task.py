"""billsum_task.py under its own task name, for the two-expert screen.

A separate name keeps this screen's logs and adapters (gemma3-billsum_split-*)
from colliding with the 2026 scaling curve's (gemma3-billsum-*), which use the
same arm names. Scoring, prompt and truncation are billsum_task.py's.
"""
from pathlib import Path

from inspect_ai import Task, task

from billsum_task import billsum

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "billsum_split" / "test"


@task
def billsum_split(prompt_variant: str = "short", dataset_path: str = str(DEFAULT_DATASET),
                  limit: int | None = None, max_tokens: int = 1280) -> Task:
    # 1280, not billsum_task.py's 640: the committed curve was re-scored at 1280
    # after the base model left 24% of generations unterminated at 640.
    return billsum(prompt_variant=prompt_variant, dataset_path=dataset_path, limit=limit, max_tokens=max_tokens)
