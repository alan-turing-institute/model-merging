"""LLM-as-a-judge for summaries, as an Inspect scorer.

The judge rates a summary against its source document on the four SummEval
dimensions (Fabbri et al., 2021), each on SummEval's 1-5 scale:

  coherence    the summary as a whole is well structured and organised, not a
               heap of related facts
  consistency  every statement is supported by the source: no invented or
               contradicted facts (hallucination is what this catches)
  fluency      each sentence is grammatical and readable, with no repetition
  relevance    it keeps the source's most important information and leaves out
               the unimportant, FOR THE LENGTH THE TASK ASKS FOR

The last clause is deliberate. DeepEval's coverage asks whether the summary
answers questions about the source, which a 20-word XSum summary cannot do for
more than one of five, so coverage there mostly measures length. Here the judge
is told what each dataset's summaries look like (STYLE) and rates relevance
within that budget.

Reference-free by default: the judge sees the source and the summary, not the
reference, so a summary is not marked down for differing from one valid
summary. `with_reference=True` shows the reference as an example of the
expected style, not as an answer key.

GROUNDED CONSISTENCY. Asked for a 1-5 consistency rating alone, Qwen3-32B gave
5 to "Actor and broadcaster Peter Lally has died aged 73." for an article that
names neither "Peter" nor the age - XSum's characteristic failure, and the one
entity_support catches. So the judge first lists every name, number, date,
place and organisation in the summary with a verbatim quote from the source
that supports it (or none). The scorer then checks two things itself, without
the judge: the quote is in the source (Source.quoted - the judge abbreviates
with "..." and joins fragments, so this is a trigram match, not exact; version 3's
exact match rejected correct facts such as "Fiji" and "YouGov survey"), and every
name and number of the fact is in the source by entity_support's rules (version 2
let the quote "Lally" support "Liam Lally"). Any fact failing either caps
consistency at 2. The facts and
their quotes are kept in the score's metadata.

LENGTH PENALTY. The prompt asks the judge to rate a summary much longer than
the task asks for lower on relevance, and it did not: the untrained 4B's
47-word XSum summaries got 4.97 against the trained models' 4.66, the verbosity
bias LLM judges are known for. So, as for consistency, the scorer applies it
itself. The task's acceptable length is the 10th-90th percentile of its own
reference word counts (length_range, computed from the test set by the caller);
relevance_len caps relevance once the summary is more than 1.5x outside it
(length_cap; the tolerance is calibrated against SummEval's experts), and
judge_mean_len uses it. The raw ratings are unchanged, so the
penalty can be applied to judgements made before it existed
(summ_metrics.py --metrics judge-len).

The judge is the `grader` model role, or `model` if given; it answers in a
JSON schema (structured output), with Qwen3's thinking switched off since a
schema-constrained answer cannot open with a <think> block. Run it on finished
evals through summ_metrics.py (--metrics judge), which re-scores the logs with
inspect_ai.score_async and keeps the judged logs for `inspect view`, or attach
it to a task's scorers like any other.
"""

import math
import re

from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field, ValidationError

from semantic_metrics import entities

DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")
# Bump with any change to the prompt or scoring: summ_metrics.py puts it in its
# file names, so judgements from different versions are never mixed or reused.
VERSION = 4

STYLE = {
    "xsum": "a single sentence of about 20 words, like the summary above a BBC News article",
    "cnndm": "three or four short highlight sentences, like those at the top of a CNN or Daily Mail article",
    # SummEval's 100 CNN/DailyMail test articles (summeval_task.py replays its systems' outputs)
    "summeval": "three or four short highlight sentences, like those at the top of a CNN or Daily Mail article",
    # LLM-AggreFact's claims (aggrefact_task.py): summaries or summary sentences
    # from recent systems; only the grounded consistency rating is read from them
    "aggrefact_cnn": "a short summary of a CNN or Daily Mail news article",
    "aggrefact_xsum": "a one-sentence summary of a BBC news article",
    "billsum": "a summary of a US bill in the style of the Congressional Research Service, usually 100-250 words",
    "billsum_split": "a summary of a US bill in the style of the Congressional Research Service, usually 100-250 words",
    "samsum": "one or two sentences in the third person saying what the participants did, decided or will do",
}

PROMPT = """You are an expert evaluator of summaries. Rate the SUMMARY of the SOURCE below.

The task asked for {style}. Judge the summary as a summary of that kind: do not
reward or penalise it for length within that budget.
{reference}
Rate each dimension from 1 (very poor) to 5 (excellent):

- coherence: the summary as a whole is well structured and well organised; it
  reads as a summary, not a heap of related facts.
- consistency: every statement in the summary is supported by the SOURCE. Any
  name, number, date or event that is not in the SOURCE, or that contradicts it,
  is an error; a summary with a fabricated fact scores at most 2.
- fluency: each sentence is grammatical and easy to read, with no repetition or
  formatting problems.
- relevance: the summary spends its length on the SOURCE's most important
  information. Do not penalise omissions the length budget forces: a
  one-sentence summary is judged on whether it states the main point. A summary
  much longer than the task asks for has included more than the most important
  content, so rate it lower.

First list, under "facts", EVERY name, number, date, place and organisation in
the SUMMARY. For each, copy a short verbatim quote from the SOURCE that states
it, or give null if the SOURCE does not state it. Do not paraphrase the quote.
Then give a brief justification (at most three sentences) and the four ratings.

[SOURCE]
{document}
[END SOURCE]

[SUMMARY]
{summary}
[END SUMMARY]
"""

REFERENCE = """
An example of the expected style (another valid summary of the same source,
not an answer key):
[EXAMPLE]
{reference}
[END EXAMPLE]
"""


_PUNCT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _words(text):
    return re.findall(r"[\w'-]+", " ".join((text or "").translate(_PUNCT).split()).lower())


def _trigrams(words):
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


class Source:
    """The document, tokenised once for every fact checked against it."""

    def __init__(self, document):
        self.document = document
        self.words = _words(document)
        self.text = " ".join(self.words)
        self.trigrams = _trigrams(self.words)

    def quoted(self, quote):
        """The quote is in the source, allowing for the judge's habits: "..."
        between fragments and the odd changed word. A fragment of three or more
        words needs 75% of its word trigrams in the source; a shorter one must
        appear as it is."""
        fragments = [_words(f) for f in re.split(r"\.\.\.|…", quote or "")]
        fragments = [f for f in fragments if f]
        for f in fragments:
            if len(f) < 3:
                if " ".join(f) not in self.text:
                    return False
            elif len(_trigrams(f) & self.trigrams) < 0.75 * len(_trigrams(f)):
                return False
        return bool(fragments)

    def names_present(self, fact):
        """Every name and number of the fact is in the source, by entity_support's
        rules (semantic_metrics.entities, case-insensitive substring), allowing a
        plural or possessive ending: "Ethiopians" is supported by "Ethiopian"."""
        doc = self.document.translate(_PUNCT).lower()
        for e in entities(fact):
            e = e.translate(_PUNCT).lower()
            if not any(c and c in doc for c in (e, e.removesuffix("'s"), e.removesuffix("s"))):
                return False
        return True

    def supports(self, fact, quote):
        return self.quoted(quote) and self.names_present(fact)


def length_range(references, lo_q=0.10, hi_q=0.90):
    """The task's acceptable summary length in words: the 10th-90th percentile
    of its references' word counts."""
    counts = sorted(len((r or "").split()) for r in references)
    pick = lambda q: counts[min(len(counts) - 1, int(q * len(counts)))]
    return pick(lo_q), pick(hi_q)


def length_cap(words, lo, hi, tolerance=1.5):
    """Highest relevance a summary of this length can get. Nothing within
    tolerance x the range; beyond it 4, past 1.5x that 3, past 2x 2 (shorter:
    4, then 3 below half). Calibrated on SummEval: capping from the range's own
    edge (tolerance 1) cut the judge's system-level agreement with the experts'
    relevance from 0.857 to 0.460 Kendall tau - the experts did not penalise
    LEAD-3's 84 words against references of 32-69 - while tolerance 1.5 keeps
    it (0.869) and still catches gross violations, such as the untrained 4B's
    205-word CNN/DailyMail summaries."""
    up, down = tolerance * hi, lo / tolerance
    if words > up:
        return 4.0 if words <= 1.5 * up else 3.0 if words <= 2 * up else 2.0
    if words < down:
        return 4.0 if words >= 0.5 * down else 3.0
    return 5.0


class Fact(BaseModel):
    fact: str
    source_quote: str | None


class Judgement(BaseModel):
    # At most 40: the judge can loop on the list, and an unbounded loop runs into
    # max_tokens and leaves unparsable JSON (2.4% of XSum and 25% of BillSum
    # judgements in the first full run, at max_tokens 1024). The schema only
    # acts at a 41st fact, so a judgement that finished before is unchanged.
    facts: list[Fact] = Field(max_length=40)
    justification: str
    coherence: int = Field(ge=1, le=5)
    consistency: int = Field(ge=1, le=5)
    fluency: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)


LENGTH_KEYS = ("words", "relevance_len", "judge_mean_len")


@scorer(metrics={d: [mean(), stderr()] for d in (*DIMENSIONS, "judge_mean", *LENGTH_KEYS)})
def summ_judge(task: str = "xsum", with_reference: bool = False, model: str | None = None,
               max_tokens: int = 4096, length_bounds: tuple[int, int] | None = None):
    """SummEval-style 1-5 ratings from an LLM judge (see module docstring)."""
    config = GenerateConfig(
        temperature=0.0, max_tokens=max_tokens,
        response_schema=ResponseSchema(name="summary_judgement", json_schema=json_schema(Judgement), strict=True),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(model) if model else get_model(role="grader", required=True)
        document = (state.metadata or {}).get("document")
        if document is None:
            raise ValueError("summ_judge needs the source in sample metadata['document']")
        prompt = PROMPT.format(
            style=STYLE[task], document=document, summary=(state.output.completion or "").strip(),
            reference=REFERENCE.format(reference=target.text) if with_reference else "")
        text = ""
        for _ in range(2):  # one retry: a constrained decode can still be cut off by max_tokens
            text = (await judge.generate(prompt, config=config)).completion
            try:
                j = Judgement.model_validate_json(text)
            except ValidationError:
                continue
            source = Source(document)
            unsupported = [f.fact for f in j.facts if not source.supports(f.fact, f.source_quote)]
            value = {d: float(getattr(j, d)) for d in DIMENSIONS}
            if unsupported:
                value["consistency"] = min(value["consistency"], 2.0)
            value["judge_mean"] = sum(value.values()) / len(DIMENSIONS)
            value.update(with_length(value, (state.output.completion or "").strip(), length_bounds))
            note = f" Unsupported: {'; '.join(unsupported)}." if unsupported else ""
            return Score(value=value, answer=state.output.completion, explanation=j.justification + note,
                         metadata={"facts": [f.model_dump() for f in j.facts], "unsupported": unsupported})
        return Score(value={d: math.nan for d in (*DIMENSIONS, "judge_mean", *LENGTH_KEYS)},
                     explanation=f"unparsable judge output: {text[:300]!r}")

    return score


def with_length(value, summary, bounds):
    """words, relevance_len and judge_mean_len for one judgement (see LENGTH
    PENALTY); without bounds the relevance is left as it is."""
    words = len(summary.split())
    cap = length_cap(words, *bounds) if bounds else 5.0
    rel = min(value["relevance"], cap)
    return {"words": float(words), "relevance_len": rel,
            "judge_mean_len": (value["coherence"] + value["consistency"] + value["fluency"] + rel) / 4}
