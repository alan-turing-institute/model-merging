"""English->German medical/legal translation as an Inspect AI task.

The task of the domain-split screen. The headline is sentence-level chrF++
(sacrebleu, word_order=2), averaged. Mean sentence chrF++ is not corpus chrF++,
but it is what a paired bootstrap over sentences needs, and the screen's
comparisons are all within this one harness. COMET is scored separately in
metrics/, whose environment it needs.

chrf_medical and chrf_law are NaN outside their domain. `truncated` is the share
of generations stopped by max_tokens rather than by the model: a translation cut
off mid-sentence scores low for a reason that has nothing to do with the model.
If it is not near zero, raise max_tokens before reading anything else.

PROMPT PARITY: the user turn is built by the same `convert_to_prompt` that
train/prepare_mt_domain_data.py uses to build the training data, and the
reference picked by the same `target`, imported rather than copied, so that
training and evaluation wording cannot drift apart. Both follow each row's
`direction`, so the German->English data (mt_domain_de_en_task.py) runs through
this task too.

Usage, from evaluate/:

    inspect eval mt_domain_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=16 -M do_sample=false -M dtype=bfloat16
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
DEFAULT_DATASET = REPO_ROOT / "datasets" / "mt_domain" / "test"
PREPARE_SCRIPT = REPO_ROOT / "train" / "prepare_mt_domain_data.py"


def load_prepare():
    """train/prepare_mt_domain_data.py, loaded by file path (its convert_to_prompt and target)."""
    spec = importlib.util.spec_from_file_location("prepare_mt_domain_data", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@scorer(metrics={"chrf": [mean(), stderr()], "chrf_medical": [mean()], "chrf_law": [mean()],
                 "truncated": [mean()], "length_ratio": [mean()]})
def chrf_pp():
    metric = CHRF(word_order=2)

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        value = metric.sentence_score(prediction, [target.text]).score
        domain = (state.metadata or {}).get("domain")
        return Score(
            value={"chrf": value,
                   "chrf_medical": value if domain == "medical" else math.nan,
                   "chrf_law": value if domain == "law" else math.nan,
                   "truncated": float(state.output.stop_reason == "max_tokens"),
                   "length_ratio": len(prediction.split()) / max(1, len(target.text.split()))},
            answer=prediction,
            explanation=f"reference: {target.text}",
        )
    return score


# Instruction wordings for the prompt-robustness check, by direction. "trained"
# is the training prompt and the default; "strict" adds an output-only
# constraint, aimed at the untrained model's explanations; "paraphrase" asks for
# the same thing in words no model was trained on; "long" spells out the task, register and output format.
VARIANTS = {
    "strict": {"en-de": "Translate this English text into German. Output only the translation, "
                        "with no explanation or notes:",
               "de-en": "Translate this German text into English. Output only the translation, "
                        "with no explanation or notes:"},
    "paraphrase": {"en-de": "Render the following passage in German.",
                   "de-en": "Give the English version of the following German passage."},
    # Detailed but domain-blind: naming medicine or law would tell a merged
    # model which expert's skill an input needs, a different experiment.
    **{"long": {d: (f"You are a professional translator of technical documents. Translate the following "
                    f"{src} text into {tgt}. Preserve the meaning exactly, keep terminology precise and "
                    f"consistent, keep numbers, units, names and formatting unchanged, and match the formal "
                    f"register of the source. Output only the {tgt} translation, with no explanation, notes "
                    f"or alternatives:")
                for d, src, tgt in (("en-de", "English", "German"), ("de-en", "German", "English"))}},
}


def user_turn(prepare, row, variant):
    """The training prompt, with its instruction swapped for the variant's wording."""
    content = prepare.convert_to_prompt(row)["messages"][0]["content"]
    if variant == "trained":
        return content
    direction = row.get("direction", "en-de")
    return content.replace(prepare.DIRECTIONS[direction][0], VARIANTS[variant][direction], 1)


@task
def mt_domain(dataset_path: str = str(DEFAULT_DATASET), limit: int | None = None,
              max_tokens: int = 512, prompt_variant: str = "trained") -> Task:
    """prompt_variant: "trained" (the default), "strict", "paraphrase" or "long" - see VARIANTS."""
    if prompt_variant not in ("trained", *VARIANTS):
        raise ValueError(f"prompt_variant must be trained, {' or '.join(VARIANTS)}; got {prompt_variant!r}")
    prepare = load_prepare()
    rows = load_from_disk(dataset_path)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    samples = [
        Sample(input=[ChatMessageUser(content=user_turn(prepare, r, prompt_variant))],
               target=prepare.target(r), id=i, metadata={"domain": r["domain"], "shard": r["shard"]})
        for i, r in enumerate(rows)
    ]
    return Task(
        dataset=MemoryDataset(samples, name="mt_domain"),
        solver=generate(),
        scorer=chrf_pp(),
        # 512 against references whose maximum the prep script prints; legal
        # sentences run long. The `truncated` metric is the check.
        config=GenerateConfig(max_tokens=max_tokens),
        version=1,
        metadata={"dataset_path": dataset_path, "prompt_variant": prompt_variant},
    )
