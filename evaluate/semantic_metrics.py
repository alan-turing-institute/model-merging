"""Semantic and grounding metrics, shared by both evaluators.

One module rather than one copy per evaluator, deliberately. This repo already
carries a metric that drifted: the crime prompt is written out in
prepare_data.py and again in evaluate.py, and the two differ by four characters
(single quotes against double), which means every crime model is trained on one
wording and tested on another. A metric duplicated across
evaluate_summarisation.py and xsum_task.py would drift the same way, and the
symptom would be two arms that look comparable and are not.

WHY THESE TWO, ON TOP OF ROUGE. ROUGE is lexical overlap, and on XSum that is a
poor proxy in both directions.

It misses correct paraphrase. Against a 21-word reference, a faithful summary
that picks different words scores near zero on ROUGE-2, so part of what ROUGE
rewards is word choice rather than meaning.

And it is far too kind to hallucination, which is XSum's characteristic failure.
Measured on this project's own outputs, for an article about the actor Mick
Lally, who died at 64:

    merge         "... Sean Lally ... has died at the age of 73."
    kd student    "... Liam Lally ... has died aged 69."

Fluent, on topic, wrong about the person and the number. ROUGE gives those about
0.4, and embedding similarity scores them HIGHER still, because as sentences
they are nearly identical to the reference. Neither notices the name was
invented. `entity_support` is the one that does.
"""

import os
import re
import statistics

_EMBEDDER = None


def embedder():
    """Load the sentence encoder once, or return None if unavailable.

    Non-fatal by design. Compute nodes have no outbound network, so a missing
    cache must degrade to "no semantic score" rather than fail a run that has
    already spent an hour generating.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer

            _EMBEDDER = SentenceTransformer(
                os.environ.get("XSUM_EMBEDDER",
                               "sentence-transformers/all-MiniLM-L6-v2")
            )
        except Exception as exc:  # noqa: BLE001 - any failure means "skip it"
            print(f"semantic metrics: embedder unavailable "
                  f"({type(exc).__name__}); reporting similarity as 0.0")
            _EMBEDDER = False
    return _EMBEDDER or None


# Capitalised words and numbers. Sentence-initial words are capitalised by
# grammar rather than by being names, so the first token of each sentence is
# dropped; numbers count wherever they appear, since "64" against "73" is
# exactly the error this exists to catch.
_ENTITY = re.compile(r"\b([A-Z][a-zA-Z'’-]+|\d[\d,.]*)\b")


def entities(text):
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        tokens = _ENTITY.findall(sentence)
        if tokens and sentence.startswith(tokens[0]):
            tokens = tokens[1:]
        found.extend(tokens)
    return found


def unsupported_entities(prediction, document):
    """Entities in the summary that do not appear in the source article.

    Case-insensitive substring, not exact token match: a summary may re-inflect
    or abbreviate a name, and demanding an exact match would score correct
    summaries as hallucinated. It is string matching rather than entity linking,
    so references themselves score below 1.0 - the number is comparative across
    models on a fixed test set, not an absolute measure of faithfulness.
    """
    haystack = (document or "").lower()
    return [e for e in entities(prediction) if e.lower() not in haystack]


def score_one(prediction, reference, document, model=None):
    """Per-example semantic similarity and entity grounding."""
    model = model if model is not None else embedder()
    similarity = 0.0
    if model is not None and prediction.strip():
        vectors = model.encode([prediction, reference], normalize_embeddings=True)
        similarity = float(vectors[0] @ vectors[1])
    found = entities(prediction)
    bad = unsupported_entities(prediction, document)
    # An entity-free prediction has nothing to ground, and scoring that 1.0
    # would let a degenerate model win this metric outright - an empty string
    # has no unsupported entities, so it would look perfectly faithful. Empty
    # output scores 0.0 because it IS a failure; a real sentence that happens to
    # name nobody is simply unmeasurable here and is excluded from the mean.
    if not prediction.strip():
        support = 0.0
    elif not found:
        support = None
    else:
        support = (len(found) - len(bad)) / len(found)
    return {
        "semantic": similarity,
        "entity_support": support,
        "entities_unsupported": float(len(bad)),
    }


def score_all(predictions, references, documents):
    """(aggregate metrics, per-example rows). Loads the encoder once."""
    model = embedder()
    rows = [
        score_one(p, r, d, model)
        for p, r, d in zip(predictions, references, documents)
    ]
    if not rows:
        return {}, rows
    metrics = {
        k: statistics.fmean(row[k] for row in rows)
        for k in ("semantic", "entities_unsupported")
    }
    gradeable = [row["entity_support"] for row in rows if row["entity_support"] is not None]
    metrics["entity_support"] = statistics.fmean(gradeable) if gradeable else float("nan")
    # Reported so the denominator is visible: a model whose summaries name
    # nobody is not being graded on grounding, and that should be obvious rather
    # than hidden inside a mean.
    metrics["num_ungradeable"] = len(rows) - len(gradeable)
    # How many summaries contain at least one invented entity - the headline
    # number, since a mean over mostly-clean output hides a tenth that is not.
    metrics["num_hallucinating"] = sum(1 for row in rows if row["entities_unsupported"])
    return metrics, rows
