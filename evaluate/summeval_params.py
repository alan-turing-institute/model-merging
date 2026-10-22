"""Do metric settings explain where our SummEval numbers differ from the paper's?

    uv run --with rouge-score --with sacrebleu --with nltk --with bert-score --with scipy \
        python summeval_params.py pairs.jsonl [--md out.md]

pairs.jsonl: one row per (article, system) from datasets/summeval/pairs, with
all 11 references. Recomputes each metric for the 16 rated systems under the
toolkit's settings and ours, and reports for each variant (a) agreement with the
paper's Table 4 values - mean absolute difference and Kendall tau over systems -
and (b) the system-level Kendall tau with the expert means, beside Table 2.

The settings, from the SummEval toolkit's code:
  ROUGE      pyrouge / ROUGE-1.5.5, 11 references            ours: rouge-score, stemmed, 1 reference
  BLEU       sacrebleu corpus BLEU, 13a, exp smoothing        ours: the same, 1 reference
  chrF       sacrebleu chrF (char order 6, beta 2, no words)  ours: the same, 1 reference
  METEOR     Java METEOR 1.5 (-l en -norm)                    ours: nltk (alpha .9, beta 3, gamma .5)
  BertScore  bert-base-uncased, layer 8, no rescaling         ours: that, and roberta-large rescaled
METEOR 1.5's English defaults are alpha .85, beta .20, gamma .60 (plus
function-word weighting and paraphrase tables, which nltk lacks); the "1.5-like"
variant sets nltk's three to those. Two systems differ in identity: the rated
M20 is GPT-2 zero-shot, Table 4's M20 row is supervised; the rated M23 is
Pegasus (C4), Table 4 lists the dynamic-mix and huge-news Pegasus. Both are left
out of the Table 4 agreement.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
DIMS = ("coherence", "consistency", "fluency", "relevance")
SPLIT = lambda t: "\n".join(x for x in re.split(r"(?<=[.!?])\s+", (t or "").strip()) if x)
TOKEN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs", type=Path)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    paper = json.loads((HERE / "summeval_paper.json").read_text())
    t4, t2 = paper["table4_full_test_set"], paper["table2_kendall_system_level_11refs"]["rows"]
    rows = [json.loads(l) for l in args.pairs.read_text().splitlines()]
    rows = [r for r in rows if r["system"] != "reference"]
    for r in rows:
        r["refs"] = json.loads(r["references"])
    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    systems = sorted(by, key=lambda s: int(s[1:]))
    human = {s: {d: statistics.fmean(r[d] for r in by[s]) for d in DIMS} for s in systems}
    comparable = [s for s in systems if s not in ("M20", "M23") and s in t4]
    variants = {}   # name -> {system: value}

    # ROUGE
    from rouge_score import rouge_scorer
    for stem in (True, False):
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=stem)
        for refs in ("1 ref", "11 refs, mean", "11 refs, max"):
            vals = {k: defaultdict(list) for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
            for s in systems:
                for r in by[s]:
                    rs = [r["reference"]] if refs == "1 ref" else r["refs"]
                    scores = [sc.score(SPLIT(x), SPLIT(r["summary"])) for x in rs]
                    agg = max if refs.endswith("max") else statistics.fmean
                    for k in vals:
                        vals[k][s].append(agg([x[k].fmeasure for x in scores]))
            for k in vals:
                variants[f"{k} | {'stem' if stem else 'no stem'} | {refs}"] = {s: statistics.fmean(v) for s, v in vals[k].items()}

    # BLEU and chrF, corpus-level as the toolkit reports them
    import sacrebleu
    for refs in ("1 ref", "11 refs"):
        b, c = {}, {}
        for s in systems:
            hyps = [r["summary"] for r in by[s]]
            streams = [[r["reference"] for r in by[s]]] if refs == "1 ref" else \
                      [[r["refs"][i] for r in by[s]] for i in range(11)]
            b[s] = sacrebleu.corpus_bleu(hyps, streams).score
            c[s] = sacrebleu.corpus_chrf(hyps, streams).score / 100
        variants[f"bleu | corpus | {refs}"] = b
        variants[f"chrf | corpus | {refs}"] = c

    # METEOR (nltk), default vs METEOR-1.5-like parameters
    import nltk
    from nltk.translate.meteor_score import meteor_score
    for res in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    tok = lambda t: [x.lower() for x in TOKEN.findall(t or "")]
    for pname, kw in (("nltk default", {}), ("METEOR-1.5-like", {"alpha": 0.85, "beta": 0.2, "gamma": 0.6})):
        for refs in ("1 ref", "11 refs"):
            v = {}
            for s in systems:
                v[s] = statistics.fmean(
                    meteor_score([tok(x) for x in ([r["reference"]] if refs == "1 ref" else r["refs"])], tok(r["summary"]), **kw)
                    for r in by[s])
            variants[f"meteor | {pname} | {refs}"] = v

    # BERTScore configurations
    from bert_score import BERTScorer
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    for model, layers, rescale, refs in (("bert-base-uncased", 8, False, "1 ref"), ("bert-base-uncased", 8, True, "1 ref"),
                                         ("bert-base-uncased", 8, False, "11 refs"), ("bert-base-uncased", 8, True, "11 refs"),
                                         ("roberta-large", 17, False, "1 ref"), ("roberta-large", 17, True, "1 ref")):
        scorer = BERTScorer(model_type=model, num_layers=layers, lang="en", rescale_with_baseline=rescale, device=device)
        v = {}
        for s in systems:
            cands = [r["summary"] for r in by[s]]
            refs_in = [r["reference"] for r in by[s]] if refs == "1 ref" else [r["refs"] for r in by[s]]
            _, _, f = scorer.score(cands, refs_in, batch_size=32)
            v[s] = float(f.mean())
        variants[f"bertscore | {model} L{layers}{' rescaled' if rescale else ''} | {refs}"] = v

    # report
    key_t4 = {"rouge1": "rouge1", "rouge2": "rouge2", "rougeL": "rougeL", "rougeLsum": "rougeL", "bleu": "bleu",
              "chrf": "chrf", "meteor": "meteor", "bertscore": "bertscore"}
    key_t2 = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rougeLsum": "ROUGE-L", "bleu": "BLEU",
              "chrf": "CHRF", "meteor": "METEOR", "bertscore": "BertScore-f"}
    out = ["| variant | Table 4: mean abs diff | Table 4: rank tau | " + " | ".join(f"experts {d[:4]} (paper)" for d in DIMS) + " |",
           "|---|---|---|" + "---|" * len(DIMS)]
    report = {}
    for name, v in variants.items():
        metric = name.split(" | ")[0]
        pk = key_t4[metric]
        mine = [v[s] for s in comparable]
        theirs = [t4[s][pk] for s in comparable]
        mad = statistics.fmean(abs(a - b) for a, b in zip(mine, theirs))
        rank = kendalltau(mine, theirs).statistic
        taus = [kendalltau([v[s] for s in systems], [human[s][d] for s in systems]).statistic for d in DIMS]
        p = t2.get(key_t2[metric], [None] * 4)
        out.append(f"| {name} | {mad:.3f} | {rank:+.2f} | " + " | ".join(
            f"{t:+.2f} ({p[i]:+.2f})" if p[i] is not None else f"{t:+.2f}" for i, t in enumerate(taus)) + " |")
        report[name] = {"per_system": v, "table4_mad": mad, "table4_rank_tau": rank, "expert_tau": dict(zip(DIMS, taus))}
    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
        args.md.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
