#!/usr/bin/env python3
"""Re-score the XSum arms with the extended metric suite.

The programme's headline on XSum is a null: no merge method and no distillation
weighting beats the merge by more than noise. A fair objection is that the null
belongs to ROUGE - a saturated lexical metric on a dataset whose gap is 0.0072 -
rather than to the models. This answers it by re-scoring the SAME stored
generations with metrics from three other families:

  cosine        L2-normalised sentence embeddings (reference-based, semantic)
  bertscore_f1  greedy token matching, baseline-rescaled (reference-based)
  nli_entail    article -> summary entailment, whole document (reference-free)
  summac_g3     the same over 3-sentence windows, max-aggregated
  geval_judge   probability-weighted LLM rating of factual support
  entity_support named entities in the summary present in the source

Nothing is regenerated: every arm's predictions are already on disk, so this is
scoring only. If the arms separate under a metric ROUGE cannot resolve, the null
is a measurement artefact. If they stay flat, the null is about the models.
"""
import argparse, glob, json, math, re, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluate"))
from semantic_metrics import embedder, entities, unsupported_entities  # noqa: E402
from rouge_score import rouge_scorer  # noqa: E402
import torch  # noqa: E402

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(t, cap):
    out = [x.strip() for x in _SENT.split(t or "") if len(x.strip()) > 10]
    return out[:cap] if out else ([t.strip()] if t and t.strip() else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--judge-model", default="google/gemma-3-4b-it")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)

    from datasets import load_from_disk
    ds = load_from_disk(a.dataset)
    col = "document" if "document" in ds.column_names else ds.column_names[0]
    docs = [ds[i][col] for i in range(min(a.limit, len(ds)))]

    sbert = embedder()
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    from bert_score import BERTScorer
    bs = BERTScorer(model_type="roberta-large", num_layers=17, lang="en",
                    rescale_with_baseline=True, device=dev)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
    nli_tok = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    nli = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-large-mnli").to(dev).eval()
    ENTAIL = [i for i, l in nli.config.id2label.items() if l.upper().startswith("ENTAIL")][0]
    jtok = AutoTokenizer.from_pretrained(a.judge_model)
    judge = AutoModelForCausalLM.from_pretrained(
        a.judge_model, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    DIGITS = {d: sorted({jtok.encode(f, add_special_tokens=False)[0]
                         for f in (str(d), " " + str(d))
                         if jtok.encode(f, add_special_tokens=False)}) for d in range(1, 6)}

    @torch.no_grad()
    def entail(doc, summ):
        enc = nli_tok(doc, summ, truncation=True, max_length=512, return_tensors="pt").to(dev)
        return float(nli(**enc).logits.softmax(-1)[0, ENTAIL])

    @torch.no_grad()
    def summac_g3(doc, summ, batch=32):
        s = sentences(doc, 40)
        hyp = sentences(summ, 8)
        if not s or not hyp:
            return None
        prem = [" ".join(s[j:j + 3]) for j in range(0, len(s), 3)]
        pairs = [(p, h) for h in hyp for p in prem]
        probs = []
        for k in range(0, len(pairs), batch):
            c = pairs[k:k + batch]
            enc = nli_tok([x[0] for x in c], [x[1] for x in c], truncation=True,
                          max_length=384, padding=True, return_tensors="pt").to(dev)
            probs.extend(nli(**enc).logits.softmax(-1)[:, ENTAIL].tolist())
        return float(st.fmean(max(probs[i * len(prem):(i + 1) * len(prem)])
                              for i in range(len(hyp))))

    @torch.no_grad()
    def geval(doc, summ):
        msg = [{"role": "user", "content":
                f"Article:\n{doc[:3000]}\n\nSummary:\n{summ}\n\n"
                "Is every fact in the summary supported by the article? "
                "Reply with one digit from 1 (badly unsupported) to 5 (fully supported)."}]
        enc = jtok.apply_chat_template(msg, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to(dev)
        lg = judge(**enc).logits[0, -1].float()
        mass = torch.tensor([torch.logsumexp(lg[i], 0) for i in DIGITS.values()])
        p = mass.softmax(0)
        return float((p * torch.arange(1, 6, dtype=p.dtype)).sum() / 5.0)

    rows = {}
    for arm in a.arms:
        f = Path(a.results_dir) / f"{arm}.json"
        if not f.is_file():
            print(f"  skip {arm} (absent)", flush=True)
            continue
        d = json.load(open(f))
        if "predictions" not in d or "rouge1" not in d.get("metrics", {}):
            print(f"  skip {arm} (not a summarisation arm with stored predictions)", flush=True)
            continue
        preds = d["predictions"][: a.limit]
        refs = d["references"][: a.limit]
        # An empty generation is a failure, not a missing value: ROUGE scores it
        # zero and entity_support scores it zero, so every metric here does the
        # same. It also has to be kept out of BERTScore, whose tokeniser path
        # for an empty string calls a method this transformers version dropped -
        # which is what killed the first attempt, on the arm with 463 empties.
        live = [i for i, x in enumerate(preds) if x.strip()]
        F = [0.0] * len(preds)
        if live:
            _, _, Fl = bs.score([preds[i] for i in live], [refs[i] for i in live])
            for j, i in enumerate(live):
                F[i] = float(Fl[j])
        n_empty = len(preds) - len(live)
        KEYS = ("rouge1", "cosine", "bertscore_f1", "nli_entail",
                "summac_g3", "geval_judge", "entity_support")
        # Aligned by example index, None where the metric does not apply, so a
        # paired bootstrap can resample articles rather than scores.
        per = {k: [None] * len(preds) for k in KEYS}
        acc = {k: [] for k in KEYS}
        for i, (p_, r_) in enumerate(zip(preds, refs)):
            def put(k, v):
                per[k][i] = v
                acc[k].append(v)

            put("rouge1", scorer.score(r_, p_)["rouge1"].fmeasure)
            put("bertscore_f1", F[i])
            if not p_.strip():
                for k in ("cosine", "nli_entail", "summac_g3", "geval_judge",
                          "entity_support"):
                    put(k, 0.0)
                continue
            if sbert is not None:
                v = sbert.encode([p_, r_], normalize_embeddings=True)
                put("cosine", float(v[0] @ v[1]))
            put("nli_entail", entail(docs[i], p_))
            sg = summac_g3(docs[i], p_)
            if sg is not None:
                put("summac_g3", sg)
            put("geval_judge", geval(docs[i], p_))
            found = entities(p_)
            if found:
                bad = unsupported_entities(p_, docs[i])
                put("entity_support", (len(found) - len(bad)) / len(found))
        rows[arm] = {"n": len(preds), "num_empty": n_empty,
                     "rouge1_stored_n1000": d["metrics"]["rouge1"],
                     **{k: (st.fmean(v) if v else None) for k, v in acc.items()},
                     "per_example": per}
        print(f"{arm:26s} " + " ".join(
            f"{k}={rows[arm][k]:.4f}" if rows[arm][k] is not None else f"{k}=n/a"
            for k in ("rouge1", "cosine", "bertscore_f1", "nli_entail", "summac_g3",
                      "geval_judge", "entity_support")), flush=True)

    # ---- bootstrap ---------------------------------------------------------
    # Every arm scores the SAME articles, so resampling articles (not scores)
    # and reusing the draws across arms gives paired intervals on the
    # difference. Those are the ones that answer "is this arm different from the
    # merge"; an independent CI on each mean is much wider and answers a
    # question nobody asked.
    import random
    rng = random.Random(0)
    BOOT, BASE = 2000, "merge"
    n_ex = max(v["n"] for v in rows.values())
    draws = [[rng.randrange(n_ex) for _ in range(n_ex)] for _ in range(BOOT)]

    def boot_mean(vals, idx):
        got = [vals[j] for j in idx if j < len(vals) and vals[j] is not None]
        return st.fmean(got) if got else None

    def ci(samples):
        s_ = sorted(x for x in samples if x is not None)
        if len(s_) < 20:
            return None
        return (s_[int(0.025 * len(s_))], s_[int(0.975 * len(s_))])

    METRICS = ("rouge1", "cosine", "bertscore_f1", "nli_entail", "summac_g3",
               "geval_judge", "entity_support")
    print(f"\n95% bootstrap intervals ({BOOT} resamples of articles); "
          f"paired difference against '{BASE}'")
    for arm, v in rows.items():
        if "per_example" not in v:
            continue
        v["ci"], v["paired_vs_" + BASE] = {}, {}
        for k in METRICS:
            vals = v["per_example"][k]
            c = ci([boot_mean(vals, idx) for idx in draws])
            if c:
                v["ci"][k] = [round(c[0], 4), round(c[1], 4)]
            if BASE in rows and arm != BASE and "per_example" in rows[BASE]:
                bvals = rows[BASE]["per_example"][k]
                diffs = []
                for idx in draws:
                    a_, b_ = boot_mean(vals, idx), boot_mean(bvals, idx)
                    if a_ is not None and b_ is not None:
                        diffs.append(a_ - b_)
                dc = ci(diffs)
                if dc:
                    v["paired_vs_" + BASE][k] = {
                        "diff": round(st.fmean(diffs), 4),
                        "ci": [round(dc[0], 4), round(dc[1], 4)],
                        "excludes_zero": bool(dc[0] > 0 or dc[1] < 0)}
        sig = [k for k, x in v["paired_vs_" + BASE].items() if x["excludes_zero"]]
        print(f"  {arm:26s} rouge1 {v['rouge1']:.4f} "
              f"CI[{v['ci'].get('rouge1', ['', ''])[0]}, {v['ci'].get('rouge1', ['', ''])[1]}]"
              + (f"   differs from {BASE} on: {', '.join(sig)}" if sig else
                 (f"   no metric differs from {BASE}" if arm != BASE else "   (baseline)")))

    # Does any metric separate the arms more than ROUGE does, relative to its own scale?
    print("\nspread across arms, as a fraction of the metric's own range over these arms")
    summary = {}
    for k in ("rouge1", "cosine", "bertscore_f1", "nli_entail", "summac_g3",
              "geval_judge", "entity_support"):
        vals = [v[k] for v in rows.values() if v.get(k) is not None]
        if len(vals) > 1:
            summary[k] = {"min": min(vals), "max": max(vals), "spread": max(vals) - min(vals),
                          "sd": st.pstdev(vals)}
            print(f"  {k:15s} min {min(vals):.4f}  max {max(vals):.4f}  "
                  f"spread {max(vals)-min(vals):.4f}  sd {st.pstdev(vals):.4f}")
    rows["_spread"] = summary
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.output, "w"), indent=2)
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
