"""SummEval's automatic metrics (Fabbri et al., 2021, TACL 9:391-409), as far as
they run without the original toolkit's pinned environment.

    from summ_extra import summeval_scores
    per_sample, corpus = summeval_scores(predictions, references, documents)

Implemented, per sample unless marked corpus:
  rouge1-4, rougeLsum  rouge-score F1 with the Porter stemmer, sentences split on
                       newlines for ROUGE-L. SummEval ran ROUGE-1.5.5 (pyrouge);
                       values differ slightly between the two implementations.
  bleu, chrf           sacrebleu, per sentence and corpus-level (Table 4 reports
                       corpus BLEU on a 0-100 scale and chrF on 0-1; so do these).
  meteor               nltk's METEOR. SummEval used the Java METEOR 1.5 through
                       pycocoevalcap, which scores lower; compare only within one.
  cider                pycocoevalcap's CIDEr, IDF over the batch's references
                       (the whole system output, as in the toolkit's batch mode).
  coverage, density,   Grusky et al. (2018): greedy extractive fragments shared
  compression          with the source; coverage = sum|f| / |S|, density =
                       sum|f|^2 / |S|, compression = |A| / |S|.
  novel_1-3            share of the summary's distinct n-grams absent from the source
  repeated_1-3         share of its distinct n-grams that occur more than once
  length               summary tokens

Not implemented: ROUGE-su*/w and ROUGE-WE, S3, MoverScore, SMS, SummaQA, BLANC
and SUPERT, whose released code pins pre-2021 libraries. BERTScore is computed by
summ_metrics.py, in both our configuration and the toolkit's default
(bert-base-uncased, layer 8, no rescaling). Tokens are regex words and
punctuation, lower-cased for the overlap statistics; SummEval used spaCy.
"""

import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def tokens(text):
    return TOKEN.findall(text or "")


def _ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def fragments(summary, article):
    """Lengths of the greedy extractive fragments (Grusky et al., 2018): from each
    summary position, the longest token run also found in the article."""
    where = defaultdict(list)
    for j, t in enumerate(article):
        where[t].append(j)
    out, i = [], 0
    while i < len(summary):
        best = 0
        for j in where.get(summary[i], ()):
            k = 0
            while i + k < len(summary) and j + k < len(article) and summary[i + k] == article[j + k]:
                k += 1
            best = max(best, k)
        if best:
            out.append(best)
            i += best
        else:
            i += 1
    return out


def data_stats(summary_text, article_text, n_max=3):
    s = [t.lower() for t in tokens(summary_text)]
    a = [t.lower() for t in tokens(article_text)]
    out = {"length": float(len(s))}
    if not s:
        return out
    f = fragments(s, a)
    out.update(coverage=sum(f) / len(s), density=sum(x * x for x in f) / len(s),
               compression=len(a) / len(s))
    for n in range(1, n_max + 1):
        grams = _ngrams(s, n)
        distinct = set(grams)
        if not distinct:
            continue
        out[f"novel_{n}"] = len(distinct - set(_ngrams(a, n))) / len(distinct)
        out[f"repeated_{n}"] = sum(1 for g, c in Counter(grams).items() if c > 1) / len(distinct)
    return out


def summeval_scores(predictions, references, documents):
    """Per-sample dicts and a corpus-level dict (BLEU, chrF, CIDEr) for one system."""
    import nltk
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    from pycocoevalcap.cider.cider import Cider
    from rouge_score import rouge_scorer

    for res in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3", "rouge4", "rougeLsum"], use_stemmer=True)
    split = lambda t: "\n".join(x for x in re.split(r"(?<=[.!?])\s+", (t or "").strip()) if x)

    per = []
    for p, r, d in zip(predictions, references, documents):
        rs = rouge.score(split(r), split(p))
        row = {k: v.fmeasure for k, v in rs.items()}
        row["bleu"] = sacrebleu.sentence_bleu(p, [r]).score
        row["chrf"] = sacrebleu.sentence_chrf(p, [r]).score / 100
        pt, rt = [t.lower() for t in tokens(p)], [t.lower() for t in tokens(r)]
        row["meteor"] = meteor_score([rt], pt) if pt and rt else 0.0
        row.update(data_stats(p, d))
        per.append(row)

    low = lambda t: " ".join(x.lower() for x in tokens(t)) or "_"
    cider, cider_each = Cider().compute_score({i: [low(r)] for i, r in enumerate(references)},
                                              {i: [low(p)] for i, p in enumerate(predictions)})
    for row, c in zip(per, cider_each):
        row["cider"] = float(c)
    corpus = {"bleu_corpus": sacrebleu.corpus_bleu(list(predictions), [list(references)]).score,
              "chrf_corpus": sacrebleu.corpus_chrf(list(predictions), [list(references)]).score / 100,
              "cider_corpus": float(cider)}
    return per, corpus
