"""German->English medical/legal translation as an Inspect AI task.

mt_domain (mt_domain_task.py) on the reverse-direction data that
train/prepare_mt_domain_data.py --direction de-en writes: the same EMEA and
JRC-Acquis pairs, the same scorer (sentence chrF++, per domain, `truncated`,
`length_ratio`), with the prompt and reference following each row's direction.
It is a task of its own so that evaluate.sbatch, run_screen.sh and
split_screen.py, which go by task name, keep its logs and adapters apart from
English->German's.

Usage, from evaluate/:

    sbatch evaluate.sbatch mt_domain_de_en google/gemma-3-4b-it
    inspect eval mt_domain_de_en_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=16 -M do_sample=false -M dtype=bfloat16
"""

from inspect_ai import Task, task

from mt_domain_task import REPO_ROOT, mt_domain

DEFAULT_DATASET = REPO_ROOT / "datasets" / "mt_domain_de_en" / "test"


@task
def mt_domain_de_en(dataset_path: str = str(DEFAULT_DATASET), limit: int | None = None,
                    max_tokens: int = 512, prompt_variant: str = "trained") -> Task:
    """mt_domain on the German->English test set."""
    return mt_domain(dataset_path=dataset_path, limit=limit, max_tokens=max_tokens,
                     prompt_variant=prompt_variant)
