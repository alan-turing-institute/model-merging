#!/usr/bin/env python3
"""Which summarisation metric notices when a summary is wrong?

Every metric here is scored on the SAME perturbations of the same references,
where the ground truth is known by construction. Two different truths are
tracked, because metrics answer two different questions:

  similar_to_reference - does it say what the reference says?  (paraphrase high)
  faithful_to_article  - is it supported by the source?        (entity swap low)

A similarity metric may legitimately score a hallucinated name high; that is not
a bug, it is the question it answers. What matters is that SOME metric in the
suite separates the faithful from the unfaithful, and this measures which.

Metrics
  rouge1          lexical overlap, F-measure, stemmed
  cosine          L2-normalised MiniLM sentence embeddings - the project's
                  `semantic`, stated explicitly as cosine on normalised vectors
  bertscore P/R/F greedy token matching over roberta-large contextual embeddings
  wmd_sim         relaxed Word Mover's Distance over MiniLM TOKEN embeddings,
                  reported as a similarity; BERTScore is a greedy relaxation of
                  the same idea, so the pair isolates matching strategy
  nli_entail      P(entailment) for article -> summary, deberta-large-mnli.
                  This is the groundedness/faithfulness family
  llm_judge       Gemma-3-4B-it asked whether every fact is supported, 1-5
  entity_support  the project's own: named entities present in the source
"""
import argparse, json, random, re, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluate"))
from semantic_metrics import embedder, entities, unsupported_entities  # noqa: E402
from rouge_score import rouge_scorer  # noqa: E402
import torch  # noqa: E402

AUX = (" is ", " was ", " are ", " were ", " has ", " have ", " had ", " will ")


def swap_entity(t, donors, rng):
    f = entities(t)
    return t.replace(f[0], rng.choice(donors), 1) if f and donors else None


def swap_number(t, rng):
    n = re.findall(r"\b\d+\b", t)
    return t.replace(n[0], str(int(rng.choice(n)) + rng.choice([7, 9, 11])), 1) if n else None


def negate(t):
    for a in AUX:
        if a in t:
            return t.replace(a, a[:-1] + " not ", 1)
    return None


def shuffle(t, rng):
    w = t.split()
    if len(w) < 4:
        return None
    rng.shuffle(w)
    return " ".join(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--judge-model", default="google/gemma-3-4b-it")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--dump-conditions", default=None,
                    help="Write the perturbation triples and exit, without loading "
                         "any model. For scoring in another environment.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)

    d = json.load(open(a.results))
    refs, preds = d["references"][: a.limit], d["predictions"][: a.limit]
    preds2 = json.load(open(a.second))["predictions"][: a.limit]
    from datasets import load_from_disk
    ds = load_from_disk(a.dataset)
    col = "document" if "document" in ds.column_names else ds.column_names[0]
    docs = [ds[i][col] for i in range(min(a.limit, len(ds)))]

    sbert = embedder()
    if sbert is None:
        raise SystemExit("no sentence encoder - the cosine column is the anchor here")
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    donors = [e for r in refs for e in entities(r)]

    # name, candidate, comparison target, says-what-reference-says, faithful-to-article
    CONDITIONS = [
        ("identity",       lambda i: refs[i],                      lambda i: refs[i],  True,  True),
        ("model_output",   lambda i: preds[i],                     lambda i: refs[i],  True,  True),
        ("paraphrase_nat", lambda i: preds[i],                     lambda i: preds2[i], True, True),
        ("entity_swap",    lambda i: swap_entity(refs[i], donors, rng), lambda i: refs[i], False, False),
        ("number_swap",    lambda i: swap_number(refs[i], rng),    lambda i: refs[i],  False, False),
        ("negation",       lambda i: negate(refs[i]),              lambda i: refs[i],  False, False),
        ("word_shuffle",   lambda i: shuffle(refs[i], rng),        lambda i: refs[i],  False, False),
        ("unrelated",      lambda i: refs[(i + 97) % len(refs)],   lambda i: refs[i],  False, False),
    ]


    if a.dump_conditions:
        # COMET lives in an isolated environment (its dependencies downgrade
        # transformers under the production evaluator), so it scores a dump
        # rather than rebuilding these perturbations. Rebuilding them would mean
        # two codepaths agreeing on an RNG call order, which is exactly the kind
        # of silent divergence this project keeps finding.
        payload = {}
        for name, cand_of, targ_of, similar, faithful in CONDITIONS:
            items = []
            for i in range(len(refs)):
                c, t = cand_of(i), targ_of(i)
                if c and t:
                    items.append({"i": i, "cand": c, "target": t, "doc": docs[i]})
            payload[name] = {"says_what_reference_says": similar,
                             "faithful_to_article": faithful, "items": items}
        Path(a.dump_conditions).parent.mkdir(parents=True, exist_ok=True)
        json.dump(payload, open(a.dump_conditions, "w"))
        print("dumped " + str(sum(len(v["items"]) for v in payload.values()))
              + " items across " + str(len(payload)) + " conditions to " + a.dump_conditions)
        return


    from bert_score import BERTScorer
    # The short name is required twice over: the bundled rescaling baseline is
    # keyed on it (rescale_baseline/en/roberta-large.tsv), and the cache is
    # aliased to it. Without rescaling, BERTScore compresses into a high narrow
    # band and scores an unrelated summary 0.84.
    bs = BERTScorer(model_type="roberta-large", num_layers=17,
                    lang="en", rescale_with_baseline=True, device=dev)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    nli_name = "microsoft/deberta-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_name)
    nli = AutoModelForSequenceClassification.from_pretrained(nli_name).to(dev).eval()
    ENTAIL = [i for i, l in nli.config.id2label.items() if l.upper().startswith("ENTAIL")][0]

    # UniEval: a T5 that answers yes/no questions about a summary, scored as
    # P(Yes) on the first decoder token. Four dimensions, and they are not
    # interchangeable - consistency asks about the document, relevance about the
    # reference, fluency about the text alone. No new dependency: it is a
    # seq2seq model under the transformers already present.
    from transformers import AutoModelForSeq2SeqLM
    uni_tok = AutoTokenizer.from_pretrained("MingZhong/unieval-sum")
    uni = AutoModelForSeq2SeqLM.from_pretrained("MingZhong/unieval-sum").to(dev).eval()
    YES = uni_tok.convert_tokens_to_ids("\u2581Yes")
    NO = uni_tok.convert_tokens_to_ids("\u2581No")
    if YES is None or YES == uni_tok.unk_token_id:
        YES, NO = uni_tok("Yes").input_ids[0], uni_tok("No").input_ids[0]
    UNI_DIMS = {
        "unieval_coh": lambda c, t, doc: f"question: Is this a coherent summary to the document? </s> summary: {c} </s> document: {doc}",
        "unieval_con": lambda c, t, doc: f"question: Is this claim consistent with the document? </s> claim: {c} </s> document: {doc}",
        "unieval_flu": lambda c, t, doc: f"question: Is this a fluent paragraph? </s> paragraph: {c}",
        "unieval_rel": lambda c, t, doc: f"question: Is this summary relevant to the reference? </s> summary: {c} </s> reference: {t}",
    }

    @torch.no_grad()
    def unieval(prompt):
        enc = uni_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
        dec = torch.tensor([[uni.config.decoder_start_token_id]], device=dev)
        lg = uni(**enc, decoder_input_ids=dec).logits[0, 0].float()
        return float(torch.softmax(torch.stack([lg[YES], lg[NO]]), 0)[0])

    judge = judge_tok = None
    if not a.no_judge:
        from transformers import AutoModelForCausalLM
        judge_tok = AutoTokenizer.from_pretrained(a.judge_model)
        judge = AutoModelForCausalLM.from_pretrained(
            a.judge_model, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()

    def cosine(x, y):
        v = sbert.encode([x, y], normalize_embeddings=True)
        return float(v[0] @ v[1])

    def wmd_sim(x, y):
        """Relaxed WMD over token embeddings, as a similarity in [0, 1]."""
        tx = sbert.encode(x.split(), normalize_embeddings=True)
        ty = sbert.encode(y.split(), normalize_embeddings=True)
        if len(tx) == 0 or len(ty) == 0:
            return 0.0
        S = torch.tensor(tx) @ torch.tensor(ty).T
        return float((S.max(dim=1).values.mean() + S.max(dim=0).values.mean()) / 2)

    @torch.no_grad()
    def entail(document, summary):
        """Naive: one pass over a truncated document. Kept for comparison."""
        enc = nli_tok(document, summary, truncation=True, max_length=512,
                      return_tensors="pt").to(dev)
        return float(nli(**enc).logits.softmax(-1)[0, ENTAIL])

    _SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")

    def sentences(text, cap):
        out = [x.strip() for x in _SENT.split(text or "") if len(x.strip()) > 10]
        return out[:cap] if out else ([text.strip()] if text and text.strip() else [])

    @torch.no_grad()
    def summac(document, summary, grain=1, batch=32):
        """SummaC-ZS: entailment of each summary sentence by its best-matching
        document sentence, averaged over summary sentences.

        The naive pass above truncates the article to 512 tokens and asks one
        question of the whole thing, which is exactly what SummaC shows to be
        weak: a summary sentence supported by paragraph nine is scored against a
        prefix that does not contain it. Here every summary sentence is paired
        with every document sentence and takes its maximum."""
        sents, hyp = sentences(document, 40), sentences(summary, 8)
        if not sents or not hyp:
            return None
        # grain>1 groups adjacent sentences: a one-sentence premise often cannot
        # entail a summary that synthesises the whole article, which is XSum's
        # defining property. Varying it separates "aggregation does not help"
        # from "the window was too small".
        prem = [" ".join(sents[j:j + grain]) for j in range(0, len(sents), grain)]
        pairs = [(p, h) for h in hyp for p in prem]
        probs = []
        for k in range(0, len(pairs), batch):
            chunk = pairs[k:k + batch]
            enc = nli_tok([c[0] for c in chunk], [c[1] for c in chunk],
                          truncation=True, max_length=384, padding=True,
                          return_tensors="pt").to(dev)
            probs.extend(nli(**enc).logits.softmax(-1)[:, ENTAIL].tolist())
        per_hyp = [max(probs[i * len(prem):(i + 1) * len(prem)]) for i in range(len(hyp))]
        return float(st.fmean(per_hyp))

    @torch.no_grad()
    def rate(document, summary):
        msg = [{"role": "user", "content":
                f"Article:\n{document[:3000]}\n\nSummary:\n{summary}\n\n"
                "Is every fact in the summary supported by the article? "
                "Reply with one digit from 1 (badly unsupported) to 5 (fully supported)."}]
        # return_dict=True is the current default and yields a BatchEncoding,
        # which generate() cannot take positionally.
        enc = judge_tok.apply_chat_template(msg, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True).to(dev)
        n_in = enc["input_ids"].shape[-1]
        out = judge.generate(**enc, max_new_tokens=4, do_sample=False,
                             pad_token_id=judge_tok.eos_token_id)
        txt = judge_tok.decode(out[0][n_in:], skip_special_tokens=True)
        m = re.search(r"[1-5]", txt)
        return int(m.group()) if m else None

    # G-Eval reads the probability the model assigns to each rating rather than
    # the digit it happens to emit. The Microsoft playbook flags numeric LLM
    # scores as unreliable - coarse, biased toward round numbers, and discarding
    # the model's own uncertainty - and this is the documented remedy.
    _DIGIT_IDS = {}

    def digit_ids():
        nonlocal_ids = {}
        for d in range(1, 6):
            ids = set()
            for form in (str(d), " " + str(d)):
                enc = judge_tok.encode(form, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            nonlocal_ids[d] = sorted(ids)
        return nonlocal_ids

    @torch.no_grad()
    def rate_geval(document, summary):
        """Expected rating under the model's own distribution over 1-5."""
        if not _DIGIT_IDS:
            _DIGIT_IDS.update(digit_ids())
        msg = [{"role": "user", "content":
                f"Article:\n{document[:3000]}\n\nSummary:\n{summary}\n\n"
                "Is every fact in the summary supported by the article? "
                "Reply with one digit from 1 (badly unsupported) to 5 (fully supported)."}]
        enc = judge_tok.apply_chat_template(msg, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True).to(dev)
        logits = judge(**enc).logits[0, -1].float()
        mass = torch.tensor([torch.logsumexp(logits[ids], 0) for ids in _DIGIT_IDS.values()])
        p = mass.softmax(0)
        return float((p * torch.arange(1, 6, dtype=p.dtype)).sum() / 5.0)

    out = {}
    for name, cand_of, targ_of, similar, faithful in CONDITIONS:
        acc = {k: [] for k in ("rouge1", "cosine", "wmd_sim", "bs_p", "bs_r", "bs_f1",
                               "nli_entail", "summac_g1", "summac_g3", "llm_judge", "geval_judge",
                               "entity_support", *UNI_DIMS)}
        cands, targs, idxs = [], [], []
        for i in range(len(refs)):
            c, t = cand_of(i), targ_of(i)
            if c and t:
                cands.append(c); targs.append(t); idxs.append(i)
        if not cands:
            continue
        P, R, F = bs.score(cands, targs)
        for j, i in enumerate(idxs):
            c, t = cands[j], targs[j]
            acc["rouge1"].append(scorer.score(t, c)["rouge1"].fmeasure)
            acc["cosine"].append(cosine(c, t))
            acc["wmd_sim"].append(wmd_sim(c, t))
            acc["bs_p"].append(float(P[j])); acc["bs_r"].append(float(R[j])); acc["bs_f1"].append(float(F[j]))
            acc["nli_entail"].append(entail(docs[i], c))
            for g, key in ((1, "summac_g1"), (3, "summac_g3")):
                v = summac(docs[i], c, grain=g)
                if v is not None:
                    acc[key].append(v)
            if judge is not None:
                r = rate(docs[i], c)
                if r is not None:
                    acc["llm_judge"].append(r / 5.0)
                acc["geval_judge"].append(rate_geval(docs[i], c))
            for key, build in UNI_DIMS.items():
                acc[key].append(unieval(build(c, t, docs[i])))
            found = entities(c)
            if found:
                bad = unsupported_entities(c, docs[i])
                acc["entity_support"].append((len(found) - len(bad)) / len(found))
        out[name] = {"n": len(cands), "says_what_reference_says": similar,
                     "faithful_to_article": faithful,
                     **{k: (st.fmean(v) if v else None) for k, v in acc.items()}}
        print(f"{name:15s} n={len(cands):4d} " +
              " ".join(f"{k}={out[name][k]:.3f}" if out[name][k] is not None else f"{k}=n/a"
                       for k in ("rouge1", "cosine", "wmd_sim", "bs_f1", "nli_entail",
                                 "summac_g3", "geval_judge", "entity_support",
                                 "unieval_con", "unieval_flu", "unieval_rel")), flush=True)

    # separation: does the metric tell the two classes apart, on each ground truth?
    sep = {}
    for key in ("rouge1", "cosine", "wmd_sim", "bs_p", "bs_f1", "nli_entail",
                "summac_g1", "summac_g3", "llm_judge", "geval_judge",
                "entity_support", "unieval_coh", "unieval_con", "unieval_flu",
                "unieval_rel"):
        for truth in ("says_what_reference_says", "faithful_to_article"):
            yes = [v[key] for k, v in out.items() if v[truth] and k != "identity" and v[key] is not None]
            no = [v[key] for v in out.values() if not v[truth] and v[key] is not None]
            if yes and no:
                sep.setdefault(key, {})[truth] = round(st.fmean(yes) - st.fmean(no), 4)
    out["separation"] = sep
    print("\nseparation (positive = the metric tells the classes apart)")
    for k, v in sep.items():
        print(f"  {k:15s} reference-similarity {v.get('says_what_reference_says', float('nan')):+.4f}"
              f"   faithfulness {v.get('faithful_to_article', float('nan')):+.4f}")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=2)
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
