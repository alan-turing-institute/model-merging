"""Faithfulness and judge scorers for the XSum task, as Inspect scorers.

WHY THIS FILE EXISTS. These measures previously lived in four standalone
scripts under experiments/ that re-scored stored JSON from the retired Isambard
pipeline. That arrangement could not read the .eval logs this repo now writes,
and it carried three defects that this file removes by construction:

  * Empty generations were scored 0.0 by one script and silently dropped by
    another, so the two result tables could not be read side by side - the
    degenerate control arm came out *above* the merge on UniEval consistency.
    Here every metric is computed in one place, per sample, and an empty
    completion scores 0.0 for all of them.
  * Articles were paired with predictions by list position, so any change of
    dataset order would have scored summaries against the wrong article in
    silence. Inspect hands each scorer its own sample; the article comes from
    state.metadata["document"] and cannot drift.
  * The G-Eval digit lookup took the first token of " 3", which on the Gemma
    tokeniser is a lone space. Every rating then shared that token's
    probability, pulling all scores toward the middle. Only ids that decode to
    the digit itself are kept now.

WHAT IT DOES NOT COVER. COMET is deliberately absent. unbabel-comet pins
transformers back to 4.x, which would downgrade the library every result in
this programme was measured under; it keeps its own environment and job (see
evaluate/comet/ and evaluate/comet_score.sbatch). It is in any case a diagnostic
rather than a ranking measure: it
rates a factually corrupted reference above a correct machine translation.

COST. Four extra models load on first use (~6 GB). Scoring is off by default;
enable per run, e.g.

    sbatch evaluate.sbatch xsum <model> -T prompt_variant=short -T faithfulness=true

A CAVEAT WORTH KEEPING IN VIEW. The G-Eval judge is gemma-3-4b-it, the same
model used as the self-distillation teacher. A judge tends to favour output
that sounds like itself, so read geval with care for students of that teacher.
"""
import math
import re

import torch
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

_SENT = re.compile(r"(?<=[.!?])\s+")
NLI_MODEL = "microsoft/deberta-large-mnli"
UNIEVAL_MODEL = "MingZhong/unieval-sum"
JUDGE_MODEL = "google/gemma-3-4b-it"
UNIEVAL_DIMS = ("consistency", "coherence", "relevance", "fluency")

_cache = {}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def sentences(text, cap=32):
    """Sentence split shared by SummaC and the length stats."""
    out = [s.strip() for s in _SENT.split(text or "") if len(s.strip()) > 10]
    return out[:cap] if out else ([text.strip()] if text and text.strip() else [])


def _bertscorer():
    if "bert" not in _cache:
        from bert_score import BERTScorer
        _cache["bert"] = BERTScorer(model_type="roberta-large", num_layers=17,
                                    lang="en", rescale_with_baseline=True,
                                    device=_device())
    return _cache["bert"]


def _nli():
    if "nli" not in _cache:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        mod = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(_device()).eval()
        entail_id = [i for i, l in mod.config.id2label.items()
                     if l.upper().startswith("ENTAIL")][0]
        _cache["nli"] = (tok, mod, entail_id)
    return _cache["nli"]


def _unieval():
    if "unieval" not in _cache:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(UNIEVAL_MODEL)
        mod = AutoModelForSeq2SeqLM.from_pretrained(UNIEVAL_MODEL).to(_device()).eval()
        yes, no = tok("Yes").input_ids[0], tok("No").input_ids[0]
        _cache["unieval"] = (tok, mod, yes, no)
    return _cache["unieval"]


def _judge():
    """Judge model plus, for each rating 1-5, the token ids that really are it."""
    if "judge" not in _cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
        mod = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL, torch_dtype=torch.bfloat16).to(_device()).eval()
        ids = {}
        for d in range(1, 6):
            keep = []
            for form in (str(d), f" {d}"):
                for tid in tok.encode(form, add_special_tokens=False):
                    # " 3" is [space, "3"] on this tokeniser. Taking the first
                    # token would give every rating the space token's mass.
                    if tok.decode([tid]).strip() == str(d) and tid not in keep:
                        keep.append(tid)
            ids[d] = keep
        _cache["judge"] = (tok, mod, ids)
    return _cache["judge"]


@torch.no_grad()
def nli_entailment(premise, hypothesis):
    tok, mod, entail_id = _nli()
    enc = tok(premise[:4000], hypothesis, return_tensors="pt",
              truncation=True, max_length=512).to(_device())
    return torch.softmax(mod(**enc).logits[0], -1)[entail_id].item()


@torch.no_grad()
def summac(document, summary):
    """Max entailment from any document sentence, averaged over summary sentences."""
    doc_s, sum_s = sentences(document), sentences(summary)
    if not doc_s or not sum_s:
        return 0.0
    return sum(max(nli_entailment(d, s) for d in doc_s) for s in sum_s) / len(sum_s)


@torch.no_grad()
def geval(document, summary):
    """G-Eval style: expected rating under the judge's own token probabilities."""
    tok, mod, ids = _judge()
    prompt = (
        "You will be given an article and a one-sentence summary of it.\n"
        "Rate how well the summary is supported by the article, from 1 to 5, "
        "where 1 is unsupported and 5 is fully supported.\n\n"
        f"Article:\n{document[:3000]}\n\nSummary:\n{summary}\n\nRating (1-5):"
    )
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(_device())
    logits = mod(**enc).logits[0, -1]
    probs = torch.softmax(logits, -1)
    mass = {d: float(sum(probs[t].item() for t in ids[d])) for d in range(1, 6)}
    total = sum(mass.values())
    if total <= 0:
        return 0.0
    expected = sum(d * m for d, m in mass.items()) / total
    return (expected - 1.0) / 4.0          # to 0-1


@torch.no_grad()
def unieval(document, summary, reference):
    """The four UniEval dimensions, each as P(Yes) on its own question."""
    tok, mod, yes_id, no_id = _unieval()
    qs = {
        "consistency": f"question: Is this claim consistent with the document? </s> claim: {summary} </s> document: {document[:3000]}",
        "coherence":   f"question: Is this a coherent summary of the document? </s> summary: {summary} </s> document: {document[:3000]}",
        "relevance":   f"question: Is this summary relevant to the reference? </s> summary: {summary} </s> reference: {reference}",
        "fluency":     f"question: Is this a fluent paragraph? </s> paragraph: {summary}",
    }
    out = {}
    for dim, q in qs.items():
        enc = tok(q, return_tensors="pt", truncation=True, max_length=1024).to(_device())
        dec = torch.tensor([[mod.config.decoder_start_token_id]]).to(_device())
        logits = mod(**enc, decoder_input_ids=dec).logits[0, -1]
        p = torch.softmax(torch.tensor([logits[yes_id], logits[no_id]]), -1)
        out[dim] = p[0].item()
    return out


@scorer(
    metrics={
        "bertscore": [mean(), stderr()],
        "nli_entail": [mean(), stderr()],
        "summac": [mean(), stderr()],
        "geval": [mean(), stderr()],
        **{f"unieval_{d}": [mean(), stderr()] for d in UNIEVAL_DIMS},
    }
)
def faithfulness():
    """Reference-free grounding measures, a judge, and UniEval's four dimensions.

    Read relevance and fluency as rankings; consistency, coherence, entailment
    and SummaC are diagnostics - each of them ranks the weakest arm in this
    programme first, because a model that reverts toward the base copies from
    the source and is therefore trivially 'faithful' to it.
    """
    def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        reference = target.text
        document = (state.metadata or {}).get("document", "")

        # An empty generation is a failure, not a missing value. Every metric
        # here scores it 0.0, so the degeneracy control behaves the same way
        # under all of them.
        if not prediction:
            value = {k: 0.0 for k in
                     ("bertscore", "nli_entail", "summac", "geval",
                      *(f"unieval_{d}" for d in UNIEVAL_DIMS))}
            return Score(value=value, answer="", explanation="empty generation scored 0.0")

        _, _, f1 = _bertscorer().score([prediction], [reference])
        value = {
            "bertscore": float(f1[0]),
            "nli_entail": nli_entailment(document, prediction),
            "summac": summac(document, prediction),
            "geval": geval(document, prediction),
        }
        for dim, v in unieval(document, prediction, reference).items():
            value[f"unieval_{dim}"] = v
        return Score(value=value, answer=prediction,
                     explanation=f"reference: {reference}")

    return score
