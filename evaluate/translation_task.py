"""Chinese-to-English translation (WMT News Commentary) as an Inspect AI task.

Scored in two stages, because the neural metrics need a different environment:

1. Generation, scored for string overlap (chrF, BLEU) and behaviour (preamble,
   notes, untranslated Chinese, pinyin, length, numbers, repetition). Run in
   this directory's environment, like xsum_task.py:

    inspect eval translation_task.py --model hf/google/gemma-3-4b-it \
      -M batch_size=8 -M do_sample=false -M dtype=bfloat16 -T prompt_variant=short

   or `sbatch evaluate.sbatch translation <model> -T prompt_variant=short`.

2. COMET-22 (source + output + reference) and CometKiwi (source + output, no
   reference) appended to that same log by `inspect score`, from comet/:

    sbatch comet_score.sbatch logs/<log>.eval

   unbabel-comet pins transformers<5 and numpy<2, so it cannot live in the
   environment that generates without changing every evaluation's stack.

WHY SO MANY METRICS. They fail in different ways, and a difference between
students is only convincing when it shows up across families:
  - chrF / BLEU: string overlap with the reference. chrF is the more reliable
    per sentence; BLEU is there for comparability with published work. Both
    are corpus-level by definition, so the headline figures are corpus_chrf and
    corpus_bleu; the per-sample chrf mean is for stderr and for `inspect view`.
  - COMET: learned from human judgements; the headline quality figure.
  - CometKiwi: needs no reference. A gold-trained student can learn the
    references' style, which reference-based metrics reward. If COMET rises and
    CometKiwi does not, the student matched the references rather than
    translating better.
  - behaviour checks: what self-distillation from a strict teacher prompt
    should actually change. Version 2, refined on the validation split from
    base-model and teacher outputs only (never students'), and now frozen for
    test: see the comment above behaviour_checks.

PROMPT PARITY: as in xsum_task.py, the prompt is built here by the same
`convert_to_prompt` that self-distill/prepare_data_NewsCommentary_zh_en.py uses
to build the training data, and `prompt_variant` must match the prompt the
model was trained on.
"""

import asyncio
import functools
import importlib.util
import math
import re
import unicodedata
from pathlib import Path

import sacrebleu
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.scorer import (
    Score, SampleScore, Target, mean, metric, scorer, stderr,
)
from inspect_ai.solver import TaskState, generate

# Resolved from this file rather than the working directory, so the default
# holds wherever `inspect eval` is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "translation" / "news_commentary_zh_en_prompts"
PREPARE_SCRIPT = REPO_ROOT / "self-distill" / "prepare_data_NewsCommentary_zh_en.py"

# prompt_variant -> the field of convert_to_prompt's output holding that chat.
PROMPT_FIELDS = {
    "short": "messages_short",
    "long": "messages_long",
    "teacher": "teacher_messages",
}


def load_convert_to_prompt():
    """convert_to_prompt from the prepare script, loaded by path because
    self-distill/ is a directory of scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location("prepare_data_nc_zh_en", PREPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_to_prompt


def to_chat(messages):
    roles = {"system": ChatMessageSystem, "user": ChatMessageUser}
    return [roles[m["role"]](content=m["content"]) for m in messages]


def translation_samples(dataset_path, split, prompt_variant, limit=None):
    # Imported here, not at module level: comet/ loads this file to score logs
    # and cannot import datasets (its pyarrow needs numpy 2; comet pins numpy<2).
    from datasets import load_from_disk

    field = PROMPT_FIELDS[prompt_variant]
    convert_to_prompt = load_convert_to_prompt()
    dataset = load_from_disk(str(dataset_path))[split]
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        Sample(
            input=to_chat(convert_to_prompt(row)[field]),
            target=row["en"],
            id=row["id"],
            # The source sentence, for the number check and for COMET.
            metadata={"source": row["zh"], "document_id": row["document_id"]},
        )
        for row in dataset
    ]


# --- string overlap ---------------------------------------------------------


def _pairs(scores: list[SampleScore]):
    predictions = [s.score.answer or "" for s in scores]
    references = [(s.score.metadata or {})["reference"] for s in scores]
    return predictions, references


@metric
def corpus_chrf():
    """chrF over the whole test set, as WMT reports it (0-100)."""

    def compute(scores: list[SampleScore]) -> float:
        predictions, references = _pairs(scores)
        return sacrebleu.corpus_chrf(predictions, [references]).score

    return compute


@metric
def corpus_bleu():
    """BLEU over the whole test set (0-100), 13a tokenisation for English."""

    def compute(scores: list[SampleScore]) -> float:
        predictions, references = _pairs(scores)
        return sacrebleu.corpus_bleu(predictions, [references]).score

    return compute


@scorer(
    metrics={
        "chrf": [corpus_chrf(), mean(), stderr()],
        # Sentence BLEU is too noisy to average; only the corpus figure is
        # reported. The per-sample value is kept for `inspect view`.
        "bleu": [corpus_bleu()],
    }
)
def overlap():
    """chrF and BLEU against the reference."""

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        return Score(
            value={
                "chrf": sacrebleu.sentence_chrf(prediction, [reference]).score,
                "bleu": sacrebleu.sentence_bleu(prediction, [reference]).score,
            },
            answer=prediction,
            explanation=f"reference: {reference}",
            # Read back by the corpus metrics.
            metadata={"reference": reference},
        )

    return score


# --- behaviour checks -------------------------------------------------------
#
# VERSION 2, frozen. Version 1 was a draft from expected failure modes; version
# 2 is refined from the 3,000 base-model validation outputs (short, long and
# teacher prompts) and the 9,978 teacher outputs in the KD training data:
#   - preamble: "Sure"/"Of course" etc. count only as a standalone
#     interjection. Every v1 hit on the long/teacher prompts was a translation
#     opening "Of course, ...", and no real preamble started that way.
#   - quoted: judged against the source, not the reference; v1's only hits were
#     sources that are themselves quotations.
#   - truncated (new): 66.5% of short-prompt outputs ran into max_tokens.
#   - alternatives (new): the short prompt's main failure, "Here are a few
#     options for translating..." followed by several candidates.
#   - notes: also catches a "breakdown" / word-by-word glossary after the
#     translation (18% of short-prompt outputs, missed by v1).
#   - echoes_source (new) / untranslated: the Chinese source quoted back in a
#     preamble, and glossary lines, no longer count as untranslated text.
#   - length_out_of_band: measured against the source, not the reference, as
#     some references are loose or misaligned; the reference ratio is kept as
#     length_ratio, for information.
#   - numbers_preserved: skips "20世纪" (20th century) and accepts spelled-out
#     tens ("two decades" for 20).
# Name errors (麦当娜 -> "McDonald's") are real but have no check: matching
# names against the reference flagged 44% of outputs, nearly all spuriously.

# Openings that introduce a translation rather than being one: "Here's the
# translation:", "Sure!", "Translation:", "English:", "**Translation**".
# "Sure"/"Of course" etc. only as an interjection ("Sure!", "Of course:",
# "Sure, here is..."), not opening a sentence ("Of course, Putin could...").
_PREAMBLE_OPENING = re.compile(
    r"^\W*(?:here(?:'|’)?s\b|here is\b|"
    r"(?:sure|okay|certainly|of course)\s*(?:[!:：]|[.,]?\s*(?:\n|$|here\b))|"
    r"(?:the |an? |my )?(?:english )?translation\b|english\s*[:：]|translated\b)",
    re.IGNORECASE,
)
# Any first line ending in a colon with more text after it.
_PREAMBLE_LINE = re.compile(r"^[^\n]*[:：][*\s]*\n\s*\S")
# Commentary after or around the translation: "Note: ...", "(Translator's
# note: ...)", "Literal translation: ...", "Explanation:", "Here's a breakdown
# of the translation:" or "word-by-word"; glossary lines are _GLOSSARY_LINE.
_NOTES = re.compile(
    r"(?:^|\n|\()\s*\**\s*(?:translator(?:'|’)?s\s+)?(?:notes?|explanation|"
    r"literal(?:ly| translation)|alternative(?: translation)?|alternatively)\s*\**\s*[:：]"
    r"|\bbreakdown\b|\bword[- ]by[- ]word\b",
    re.IGNORECASE,
)
# A glossary line: "* **有鉴于此 (yǒu jiàn yú cǐ):** Given this".
_GLOSSARY_LINE = re.compile(
    r"^[\s*•\-]*\**[㐀-䶿一-鿿豈-﫿]+\**\s*[(（][^)）\n]*[)）]\s*\**\s*[:：]",
    re.MULTILINE,
)
# Several candidate translations instead of one: "Here are a few options for
# translating...", "Option 1:", "Version 2".
_ALTERNATIVES = re.compile(
    r"\b(?:a few|several|some|two|three|four|multiple)\s+(?:different\s+)?"
    r"(?:ways|options|possible translations|translations|versions)\b"
    r"|\b(?:option|version)\s*\d",
    re.IGNORECASE,
)
_MARKDOWN = re.compile(r"\*\*|__|^#{1,6}\s|^\s*(?:[-*•]|\d+\.)\s", re.MULTILINE)
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿　-〿！-｠]")
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# A run of Chinese this long that also occurs in the source is the source
# quoted back ('Here are a few ways to translate "然而，正如智者所言…"'), not a
# word left untranslated.
_CJK_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿　-〿！-｠]{8,}")
# Pinyin tone marks that English (and European names) never use: macrons and
# carons. á, é, à etc. are left out - they are common in names like "Macron's
# Élysée" - so second- and fourth-tone-only pinyin is missed.
_PINYIN = re.compile(r"[āēīōūǖǎěǐǒǔǚǘǜ]", re.IGNORECASE)
_QUOTED = re.compile(r"^\s*[\"“「『'‘].*[\"”」』'’]\s*$", re.DOTALL)
# English words per Chinese character of the source: outside this band the
# output has probably dropped or added content. The 1st-99th percentile of the
# training references (median 0.59), so ~2% of professional translations fall
# outside it too.
LENGTH_BAND = (0.35, 0.95)
# Gemma 3 tokens, the generation limit; an output within a few tokens of it was
# cut off (re-tokenising the text can differ from the generated ids by one or
# two). The longest long/teacher-prompt output is 103 tokens.
MAX_TOKENS = 128
TRUNCATED_TOKENS = MAX_TOKENS - 3

# The same word twice in a row, as in xsum_task.py (copied rather than
# imported: importing a task file would register its task here too).
_REPEATED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
_REPEATED_RUN = re.compile(r"\b(\w+)(?:\s+\1\b){3,}", re.IGNORECASE)

# Arabic numerals, with thousands separators or decimals.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Chinese words after a number that stop its digits carrying over: magnitudes
# ("300万" is "3 million") and centuries ("20世纪" is "the 20th century").
_MAGNITUDE = ("万", "亿", "千", "百", "十", "世纪")
# Spelled-out forms accepted for round numbers: "two decades" or "twenty" for 20.
_TENS = {
    "10": ("ten", "a decade"), "20": ("twenty", "two decades"),
    "30": ("thirty", "three decades"), "40": ("forty", "four decades"),
    "50": ("fifty", "five decades"), "60": ("sixty", "six decades"),
    "70": ("seventy", "seven decades"), "80": ("eighty", "eight decades"),
    "90": ("ninety", "nine decades"), "100": ("a hundred", "one hundred"),
}


def _numbers(text: str, magnitude_free: bool = False) -> set[str]:
    """Numbers in text, normalised ("2,000" -> "2000"). Single digits are
    skipped: English often spells them out ("five")."""
    text = unicodedata.normalize("NFKC", text)  # full-width digits
    found = set()
    for match in _NUMBER.finditer(text):
        if magnitude_free and text[match.end():].startswith(_MAGNITUDE):
            continue
        number = match.group(0).replace(",", "").rstrip(".")
        if len(number) > 1:
            found.add(number)
    return found


def numbers_preserved(prediction: str, source: str, reference: str):
    """Share of the source's numbers that the output keeps, or None.

    Only numbers that the reference ALSO keeps verbatim are graded, so a number
    the professional translator legitimately rewrote is not held against the
    model.
    """
    graded = _numbers(source, magnitude_free=True) & _numbers(reference)
    if not graded:
        return None
    found = _numbers(prediction)
    lowered = prediction.lower()
    kept = [n for n in graded if n in found or any(w in lowered for w in _TENS.get(n, ()))]
    return len(kept) / len(graded)


@functools.cache
def _tokenizer():
    """Gemma 3's tokenizer, shared by every model evaluated here (all Gemma 3)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/gemma-3-4b-it")


def _echoed_runs(prediction: str, source: str) -> list[str]:
    """Runs of Chinese in the output that are the source quoted back."""
    return [run for run in _CJK_RUN.findall(prediction) if run in source]


def _untranslated_cjk(prediction: str, source: str) -> int:
    """Chinese characters left in the output, not counting the source quoted
    back or glossary lines."""
    text = _GLOSSARY_LINE.sub("", prediction)
    for run in _echoed_runs(text, source):
        text = text.replace(run, "")
    return len(_CJK.findall(text))


@scorer(
    metrics={
        # Shares of outputs, so the mean is the fraction showing each behaviour.
        "preamble": [mean(), stderr()],
        "alternatives": [mean(), stderr()],
        "notes": [mean(), stderr()],
        "multi_line": [mean(), stderr()],
        "markdown": [mean(), stderr()],
        "quoted": [mean(), stderr()],
        "untranslated": [mean(), stderr()],
        "echoes_source": [mean(), stderr()],
        "pinyin": [mean(), stderr()],
        "truncated": [mean(), stderr()],
        "length_out_of_band": [mean(), stderr()],
        "repeated_word": [mean(), stderr()],
        "repeated_run": [mean()],
        # Not shares.
        "cjk_share": [mean()],
        "source_length_ratio": [mean(), stderr()],
        "length_ratio": [mean(), stderr()],
        "numbers_preserved": [mean(), stderr()],
        "numbers_gradeable": [mean()],
    }
)
def behaviour_checks():
    """Whether each output is a clean, complete English translation and nothing
    else. Counted on the whole completion, preamble included."""
    low, high = LENGTH_BAND

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        source = (state.metadata or {}).get("source", "")
        words = len(prediction.split())
        ratio = words / max(len(reference.split()), 1)
        source_ratio = words / max(len(_HAN.findall(source)), 1)
        chars = [c for c in prediction if not c.isspace()]
        cjk = len(_CJK.findall(prediction))
        tokens = len(_tokenizer()(prediction, add_special_tokens=False).input_ids)
        # Repeats the reference has too ("had had") are not counted.
        repeats = [
            m.group(0) for m in _REPEATED_WORD.finditer(prediction)
            if m.group(0).lower() not in reference.lower()
        ]
        numbers = numbers_preserved(prediction, source, reference)
        value = {
            "preamble": float(bool(
                _PREAMBLE_OPENING.match(prediction) or _PREAMBLE_LINE.match(prediction)
            )),
            "alternatives": float(bool(_ALTERNATIVES.search(prediction))),
            "notes": float(bool(
                _NOTES.search(prediction) or _GLOSSARY_LINE.search(prediction)
            )),
            "multi_line": float(len([l for l in prediction.splitlines() if l.strip()]) > 1),
            "markdown": float(bool(_MARKDOWN.search(prediction))),
            # Only when the source is not itself one quotation.
            "quoted": float(bool(_QUOTED.match(prediction) and not _QUOTED.match(source))),
            "untranslated": float(_untranslated_cjk(prediction, source) > 0),
            "echoes_source": float(bool(_echoed_runs(prediction, source))),
            "pinyin": float(bool(_PINYIN.search(prediction))),
            "truncated": float(tokens >= TRUNCATED_TOKENS),
            "length_out_of_band": float(not low <= source_ratio <= high),
            "repeated_word": float(bool(repeats)),
            "repeated_run": float(bool(_REPEATED_RUN.search(prediction))),
            "cjk_share": cjk / len(chars) if chars else 0.0,
            "source_length_ratio": source_ratio,
            "length_ratio": ratio,
            # NaN is Inspect's "unscored" marker for a key of a dict-valued
            # score: a sentence with no gradeable numbers is left out of the
            # mean, and numbers_gradeable keeps the denominator visible.
            "numbers_preserved": math.nan if numbers is None else numbers,
            "numbers_gradeable": 0.0 if numbers is None else 1.0,
        }
        return Score(
            value=value,
            answer=prediction,
            explanation=f"words per source character {source_ratio:.2f}, "
            f"length ratio to reference {ratio:.2f}, {tokens} tokens"
            + (f" | repeated: {', '.join(repeats)}" if repeats else ""),
        )

    return score


# --- COMET (comet/ environment only) ----------------------------------------
#
# Not in the task's scorer list: run through `inspect score` from comet/, see
# comet_score.sbatch. Inspect scores samples concurrently, one call each, so the
# calls are gathered into batches here - one COMET predict per sample would
# spend most of its time building PyTorch Lightning trainers.

COMET_MODEL = "Unbabel/wmt22-comet-da"
COMETKIWI_MODEL = "Unbabel/wmt22-cometkiwi-da"
COMET_BATCH = 64
# How long a partial batch waits for more samples before it runs anyway.
COMET_WAIT = 0.5
# One predict at a time, across both models: each predict builds a Lightning
# trainer whose teardown moves the model back to the CPU, so two overlapping
# predicts on one model leave the later one with a CPU model and CUDA inputs.
_PREDICT_LOCK = asyncio.Lock()


class _CometBatcher:
    def __init__(self, model_name: str):
        import logging

        from comet import download_model, load_from_checkpoint

        # Lightning logs its GPU/TPU banner on every predict call.
        for name in ("lightning.pytorch", "pytorch_lightning", "lightning"):
            logging.getLogger(name).setLevel(logging.ERROR)
        self.model = load_from_checkpoint(download_model(model_name))
        self.pending: list[tuple[dict, asyncio.Future]] = []
        self.timer: asyncio.TimerHandle | None = None

    async def score(self, item: dict) -> float:
        future = asyncio.get_running_loop().create_future()
        self.pending.append((item, future))
        if len(self.pending) >= COMET_BATCH:
            await self._flush()
        elif self.timer is None:
            self.timer = asyncio.get_running_loop().call_later(
                COMET_WAIT, lambda: asyncio.ensure_future(self._flush())
            )
        return await future

    async def _flush(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        batch, self.pending = self.pending, []
        if not batch:
            return
        try:
            import torch

            async with _PREDICT_LOCK:
                output = await asyncio.to_thread(
                    self.model.predict,
                    [item for item, _ in batch],
                    batch_size=COMET_BATCH,
                    gpus=1 if torch.cuda.is_available() else 0,
                    progress_bar=False,
                    # COMET defaults to forked DataLoader workers; forking from
                    # this worker thread is refused by filelock ("os.fork is
                    # unsafe"). The batch is already in memory, so workers gain
                    # nothing.
                    num_workers=0,
                )
            for (_, future), value in zip(batch, output.scores):
                future.set_result(float(value))
        except Exception as exc:  # noqa: BLE001 - surface it on every sample
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)


_BATCHERS: dict[str, _CometBatcher] = {}


def _batcher(model_name: str) -> _CometBatcher:
    if model_name not in _BATCHERS:
        _BATCHERS[model_name] = _CometBatcher(model_name)
    return _BATCHERS[model_name]


@scorer(
    metrics={
        "comet": [mean(), stderr()],
        "cometkiwi": [mean(), stderr()],
    }
)
def comet(kiwi: bool = True):
    """COMET-22 with the reference, and CometKiwi without it (0-1, higher is
    better). kiwi=False skips CometKiwi, which is gated on Hugging Face."""

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        source = (state.metadata or {})["source"]
        value = {
            "comet": await _batcher(COMET_MODEL).score(
                {"src": source, "mt": prediction, "ref": target.text}
            ),
            "cometkiwi": (
                await _batcher(COMETKIWI_MODEL).score({"src": source, "mt": prediction})
                if kiwi else math.nan
            ),
        }
        return Score(value=value, answer=prediction)

    return score


@task
def translation(
    prompt_variant: str = "short",
    dataset_path: str = str(DEFAULT_DATASET),
    split: str = "test",
    limit: int | None = None,
) -> Task:
    """WMT News Commentary zh->en, scored by chrF, BLEU and behaviour checks.

    prompt_variant: which prepare_data_NewsCommentary_zh_en.py prompt to
    evaluate with - "short", "long" or "teacher". Must match what the model was
    trained on.
    split: "test" for reported figures; "validation" is where the behaviour
    checks were developed.
    """
    if prompt_variant not in PROMPT_FIELDS:
        raise ValueError(
            f"prompt_variant must be one of {sorted(PROMPT_FIELDS)}, "
            f"got {prompt_variant!r}"
        )

    return Task(
        dataset=MemoryDataset(
            translation_samples(dataset_path, split, prompt_variant, limit),
            name=f"news_commentary_zh_en_{split}",
        ),
        solver=generate(),
        scorer=[overlap(), behaviour_checks()],
        # English references reach 113 Gemma tokens at most (mean 28.5), and
        # the KD data was generated with max_tokens 128.
        config=GenerateConfig(max_tokens=MAX_TOKENS),
        # 1: first draft of the behaviour checks.
        # 2: behaviour checks refined on validation outputs and frozen (see the
        #    comment above behaviour_checks).
        version=2,
        # Kept in the log, so logs rescored with a later version of the checks
        # (whose task version stays as generated) can be told apart.
        metadata={"prompt_variant": prompt_variant, "split": split, "behaviour_checks_version": 2},
    )
