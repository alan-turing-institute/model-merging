"""SummEval meta-evaluation and comparison (Fabbri et al., 2021, TACL).

    uv run python summeval_meta.py [--root ../results/summ_metrics_summeval] [--md ../results/summeval-comparison.md]

Inputs: summ_metrics.py's per-sample records for SummEval's 16 rated systems
(task summeval, replayed by summeval_task.py) and for our CNN/DailyMail models
on the same 100 articles (task cnndm, dataset summeval/cnndm_test), both under
--root; the expert ratings in datasets/summeval/pairs; the paper's Tables 2-4
in summeval_paper.json.

Reports:
  1. the expert means recomputed from the release against the paper's Table 3
     (a check that the data were joined correctly)
  2. system-level Kendall tau between each metric and the expert means over the
     16 systems - the paper's Table 2 protocol - beside the paper's values. The
     paper scored against 11 references per article; these use the original one
  3. summary-level Kendall tau: per article across the 16 systems, averaged
  4. our models placed among SummEval's systems on the same 100 articles
  5. the paper's Table 4 (full test set) beside our recomputation on the 100
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from datasets import load_from_disk
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
DIMS = ("coherence", "consistency", "fluency", "relevance")
JUDGE, BS, SE = "judge4-qwen3-32b", "bertscore-roberta-large", "summeval-toolkit"
DE, CM = "deepeval-qwen3-32b-n5", "comet-wmt22"  # DeepEval SummarizationMetric; COMET-22 (summ_comet.py)
# (column, record tag, key in the record, paper Table 2 row)
METRICS = [
    ("judge coherence", JUDGE, "coherence", None), ("judge consistency", JUDGE, "consistency", None),
    ("judge fluency", JUDGE, "fluency", None), ("judge relevance", JUDGE, "relevance", None),
    ("judge mean", JUDGE, "judge_mean", None),
    ("DeepEval score", DE, "deepeval", None), ("DeepEval alignment", DE, "alignment", None),
    ("DeepEval coverage", DE, "coverage", None), ("COMET-22", CM, "comet", None),
    ("ROUGE-1", SE, "rouge1", "ROUGE-1"), ("ROUGE-2", SE, "rouge2", "ROUGE-2"), ("ROUGE-3", SE, "rouge3", "ROUGE-3"),
    ("ROUGE-4", SE, "rouge4", "ROUGE-4"), ("ROUGE-L", SE, "rougeLsum", "ROUGE-L"),
    ("BertScore-p (toolkit)", SE, "bertscore_se_p", "BertScore-p"), ("BertScore-r (toolkit)", SE, "bertscore_se_r", "BertScore-r"),
    ("BertScore-f (toolkit)", SE, "bertscore_se_f1", "BertScore-f"), ("BERTScore F1 (ours)", BS, "bertscore_f1", None),
    ("BLEU", SE, "bleu", "BLEU"), ("CHRF", SE, "chrf", "CHRF"), ("CIDEr", SE, "cider", "CIDEr"), ("METEOR", SE, "meteor", "METEOR"),
    ("Length", SE, "length", "Length"), ("Novel unigram", SE, "novel_1", "Novel unigram"),
    ("Novel bi-gram", SE, "novel_2", "Novel bi-gram"), ("Novel tri-gram", SE, "novel_3", "Novel tri-gram"),
    ("Repeated unigram", SE, "repeated_1", "Repeated unigram"), ("Repeated bi-gram", SE, "repeated_2", "Repeated bi-gram"),
    ("Repeated tri-gram", SE, "repeated_3", "Repeated tri-gram"), ("Stats-coverage", SE, "coverage", "Stats-coverage"),
    ("Stats-compression", SE, "compression", "Stats-compression"), ("Stats-density", SE, "density", "Stats-density"),
]
OURS = [("untrained 4B", ["base-gemma-3-4b-it"]), ("reference-trained, 500 rows", ["gemma3-cnndm_curve-n500-lora"]),
        ("reference-trained, 8,000 rows", ["gemma3-cnndm_curve-n8000-lora"]),
        ("gold KD full (500 rows)", ["gemma3-cnndm-kdgold75_full-lora"]),
        ("gold KD halves (250 rows, 6)", [f"gemma3-cnndm-kdgold75_{h}{s}-lora" for h in "ab" for s in ("", "_s2", "_s3")])] + \
       [(f"merged: {m} (3 seeds)", [f"cnndm-kdgold75-merged-{m}{s}" for s in ("", "_s2", "_s3")])
        for m in ("linear", "ties", "dare_ties", "task_arithmetic")]


def records(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("error") is None:
                out[r["id"]] = r
    return out


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=HERE.parent / "results" / "summ_metrics_summeval")
    ap.add_argument("--pairs", type=Path, default=HERE.parent / "datasets" / "summeval" / "pairs")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    paper = json.loads((HERE / "summeval_paper.json").read_text())
    pairs = load_from_disk(str(args.pairs))
    human = {(r["id"], r["system"]): {d: r[d] for d in DIMS} for r in pairs if r["system"] != "reference"}
    rated = sorted({s for _, s in human}, key=lambda s: int(s[1:]))
    systems = rated + ["reference"]
    recs = {s: {tag: records(args.root / "summeval" / f"summeval-{s}.{tag}.jsonl") for tag in (JUDGE, BS, SE, DE, CM)}
            for s in systems}
    corpus = json.loads((args.root / "summeval" / "summary.json").read_text())
    out = []

    def value(s, tag, key, article):
        r = recs[s][tag].get(article)
        return None if r is None else r.get(key)

    def sysmean(s, tag, key):
        vals = [v for v in (r.get(key) for r in recs[s][tag].values()) if v is not None]
        return statistics.fmean(vals) if vals else None

    def hmean(s, d):
        return statistics.fmean(v[d] for (a, sy), v in human.items() if sy == s)

    # 1. data check
    t3 = paper["table3_expert_means"]["rows"]
    out += ["## 1. Expert means from the release vs the paper's Table 3", "",
            "| system | ours (coh / con / flu / rel) | paper | max abs diff |", "|---|---|---|---|"]
    for s in rated:
        mine = [hmean(s, d) for d in DIMS]
        cands = {k: v for k, v in t3.items() if k == s or k.startswith(s + " ")}
        best = min(cands.items(), key=lambda kv: max(abs(a - b) for a, b in zip(mine, kv[1]["scores"]))) if cands else None
        if best:
            diff = max(abs(a - b) for a, b in zip(mine, best[1]["scores"]))
            out.append(f"| {s} {best[1]['name']} | {' / '.join(fmt(x, 2) for x in mine)} | "
                       f"{' / '.join(fmt(x, 2) for x in best[1]['scores'])} | {diff:.3f} |")

    # 2. system-level Kendall tau, 16 systems
    t2 = paper["table2_kendall_system_level_11refs"]["rows"]
    out += ["", "## 2. System-level Kendall tau with expert means (16 systems)", "",
            "Ours: original reference only. Paper (Table 2): 11 references per article.", "",
            "| metric | " + " | ".join(f"{d} (ours / paper)" for d in DIMS) + " |", "|---|" + "---|" * len(DIMS)]
    for name, tag, key, prow in METRICS:
        xs = [sysmean(s, tag, key) for s in rated]
        if any(x is None for x in xs):
            continue
        cells = []
        for i, d in enumerate(DIMS):
            tau = kendalltau(xs, [hmean(s, d) for s in rated]).statistic
            p = t2.get(prow, [None] * 4)[i] if prow else None
            cells.append(f"{fmt(tau)} / {fmt(p)}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    # 3. summary-level Kendall tau, averaged over articles
    articles = sorted({a for a, _ in human})
    out += ["", "## 3. Summary-level Kendall tau (per article across the 16 systems, averaged)", "",
            "| metric | " + " | ".join(DIMS) + " |", "|---|" + "---|" * len(DIMS)]
    for name, tag, key, _ in METRICS:
        cells = []
        for d in DIMS:
            taus = []
            for a in articles:
                xs = [value(s, tag, key, a) for s in rated]
                ys = [human[(a, s)][d] for s in rated]
                if any(x is None for x in xs):
                    continue
                t = kendalltau(xs, ys).statistic
                if not math.isnan(t):
                    taus.append(t)
            cells.append(fmt(statistics.fmean(taus)) if taus else "-")
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    # 4. placement on the same 100 articles
    cols = [("judge coh", JUDGE, "coherence"), ("judge con", JUDGE, "consistency"), ("judge flu", JUDGE, "fluency"),
            ("judge rel", JUDGE, "relevance"), ("ROUGE-1", SE, "rouge1"), ("ROUGE-2", SE, "rouge2"),
            ("ROUGE-L", SE, "rougeLsum"), ("BS-f toolkit", SE, "bertscore_se_f1"), ("BS F1 ours", BS, "bertscore_f1"),
            ("METEOR", SE, "meteor"), ("length", SE, "length"), ("coverage", SE, "coverage"), ("density", SE, "density"),
            ("novel 2-gram", SE, "novel_2")]
    out += ["", "## 4. SummEval's systems and our models on the same 100 articles", "",
            "Expert columns are the paper's annotators (SummEval systems only); every other column is computed here.", "",
            "| system | experts coh / con / flu / rel | " + " | ".join(c[0] for c in cols) + " |",
            "|---|---|" + "---|" * len(cols)]
    names = {k: v["name"] for k, v in t3.items()}
    for s in systems:
        h = " / ".join(fmt(hmean(s, d), 2) for d in DIMS) if s in rated else "-"
        out.append(f"| {s} {names.get(s, 'reference' if s == 'reference' else '')} | {h} | "
                   + " | ".join(fmt(sysmean(s, tag, key), 2 if tag == JUDGE else 3) for _, tag, key in cols) + " |")
    for title, labels in OURS:
        per = []
        for c in cols:
            vals = []
            for lab in labels:
                r = records(args.root / "cnndm" / f"{lab}.{c[1]}.jsonl")
                v = [x[c[2]] for x in r.values() if x.get(c[2]) is not None]
                if v:
                    vals.append(statistics.fmean(v))
            per.append(statistics.fmean(vals) if vals else None)
        if any(p is not None for p in per):
            out.append(f"| **ours: {title}** | - | " + " | ".join(fmt(p, 2 if c[1] == JUDGE else 3) for p, c in zip(per, cols)) + " |")

    # 5. paper Table 4 (full test set) vs recomputed on the 100 articles
    t4 = paper["table4_full_test_set"]
    keys = [("ROUGE-1", "rouge1", "rouge1"), ("ROUGE-2", "rouge2", "rouge2"), ("ROUGE-L", "rougeL", "rougeLsum"),
            ("BertScore", "bertscore", "bertscore_se_f1"), ("BLEU", "bleu", "bleu_corpus"), ("CHRF", "chrf", "chrf_corpus"),
            ("METEOR", "meteor", "meteor"), ("Length", "length", "length"), ("density", "density", "density")]
    out += ["", "## 5. Paper's Table 4 (full CNN/DM test set) vs recomputed here (100 annotated articles)", "",
            "Each cell: paper / here. BLEU and CHRF here are corpus-level over the 100; METEOR is nltk's, not the Java METEOR 1.5.", "",
            "| system | " + " | ".join(k[0] for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for s in rated:
        prow = t4.get(s) or next((v for k, v in t4.items() if k.startswith(s + " ")), None)
        if not prow:
            continue
        entry = (corpus.get(f"summeval-{s}") or {}).get(SE, {})
        cells = [f"{fmt(prow.get(pk), 3)} / {fmt(entry.get(ok), 3)}" for _, pk, ok in keys]
        out.append(f"| {s} {prow.get('name', '')} | " + " | ".join(cells) + " |")

    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.write_text("# SummEval comparison (Fabbri et al., 2021)\n\n" + text + "\n")


if __name__ == "__main__":
    main()
