"""Re-score the stored XSum generations under UniEval's four dimensions.

WHY THIS EXISTS. The metric validation ranked nine measures by their ability to
separate an intact reference summary from a deliberately corrupted one, and
UniEval won by a factor of two:

    UniEval consistency  +0.562      LLM judge (G-Eval)  +0.308
    UniEval coherence    +0.555      entity support      +0.268
    UniEval relevance    +0.471      NLI entailment      +0.266
                                     BERTScore           -0.003
                                     cosine              -0.012
                                     ROUGE-1             -0.249

UniEval was then never applied to the arms. rescore_xsum.py covers rouge1,
cosine, bertscore_f1, nli_entail, summac_g3, geval_judge and entity_support -
which includes the three weakest members of that list and excludes the four
strongest. So the XSum null currently rests on a battery whose primary metric
cannot tell a corrupted summary from an intact one.

This closes that. It regenerates nothing: every arm's predictions are already on
disk, so the cost is four forward passes of a 770M seq2seq model per example.

WHAT WOULD CHANGE THE CONCLUSION. If no healthy arm separates from the merge
under any dimension, the null is as strong as this programme can make it. If one
does, that is a difference seven other metrics could not see - and the
kd-aligned arm, which already separates on four of seven in opposite directions
(worse ROUGE-1 and BERTScore, better SummaC and entity support), is the obvious
candidate.

The bootstrap is deliberately identical to rescore_xsum.py's - same seed, same
2,000 article resamples, same 'merge' baseline - so the two tables can be read
side by side.
"""

import argparse
import glob
import json
import os
import statistics as st

import torch

# Prompt templates and the Yes/No read-out are copied verbatim from
# metric_bakeoff.py, so the numbers here are on the same scale as the
# perturbation study that motivated using UniEval at all.
TEMPLATES = {
    "unieval_coh": lambda c, t, doc: f"question: Is this a coherent summary to the document? </s> summary: {c} </s> document: {doc}",
    "unieval_con": lambda c, t, doc: f"question: Is this claim consistent with the document? </s> claim: {c} </s> document: {doc}",
    "unieval_flu": lambda c, t, doc: f"question: Is this a fluent paragraph? </s> paragraph: {c}",
    "unieval_rel": lambda c, t, doc: f"question: Is this summary relevant to the reference? </s> summary: {c} </s> reference: {t}",
}
METRICS = tuple(TEMPLATES)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)

    from datasets import load_from_disk
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ds = load_from_disk(a.dataset)
    col = "document" if "document" in ds.column_names else ds.column_names[0]
    docs = [r[col] for r in ds.select(range(min(a.limit, len(ds))))]

    tok = AutoTokenizer.from_pretrained("MingZhong/unieval-sum")
    uni = AutoModelForSeq2SeqLM.from_pretrained("MingZhong/unieval-sum").to(dev).eval()
    YES = tok.convert_tokens_to_ids("▁Yes")
    NO = tok.convert_tokens_to_ids("▁No")
    if YES is None or YES == tok.unk_token_id:
        YES, NO = tok("Yes").input_ids[0], tok("No").input_ids[0]

    @torch.no_grad()
    def unieval(prompt):
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
        dec = torch.tensor([[uni.config.decoder_start_token_id]], device=dev)
        lg = uni(**enc, decoder_input_ids=dec).logits[0, 0].float()
        return float(torch.softmax(torch.stack([lg[YES], lg[NO]]), 0)[0])

    rows = {}
    for arm in a.arms:
        matches = sorted(glob.glob(os.path.join(a.results_dir, f"{arm}.json")))
        if not matches:
            print(f"  skip {arm} (no result file)", flush=True)
            continue
        d = json.load(open(matches[0]))
        if "predictions" not in d or "rouge1" not in d.get("metrics", {}):
            print(f"  skip {arm} (not a summarisation arm with stored predictions)", flush=True)
            continue
        preds = d["predictions"][: a.limit]
        refs = d["references"][: a.limit]

        # Aligned by index, with None where an arm produced nothing to score, so
        # the paired bootstrap below compares the same article across arms.
        per = {k: [None] * len(preds) for k in METRICS}
        for i, (p, r) in enumerate(zip(preds, refs)):
            if not (p or "").strip():
                continue
            doc = docs[i] if i < len(docs) else ""
            for k, build in TEMPLATES.items():
                per[k][i] = unieval(build(p, r, doc))
        row = {"n": len(preds), "per_example": per}
        for k in METRICS:
            got = [x for x in per[k] if x is not None]
            row[k] = round(st.fmean(got), 4) if got else None
        rows[arm] = row
        print(f"  {arm:26s} " + "  ".join(f"{k.split('_')[1]} {row[k]}" for k in METRICS),
              flush=True)

    # ---- bootstrap, identical to rescore_xsum.py --------------------------
    import random
    rng = random.Random(0)
    BOOT, BASE = 2000, "merge"
    if not rows:
        raise SystemExit("no arms scored")
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

    print(f"\n95% bootstrap intervals ({BOOT} resamples of articles); "
          f"paired difference against '{BASE}'")
    for arm, v in rows.items():
        v["ci"], v["paired_vs_" + BASE] = {}, {}
        for k in METRICS:
            vals = v["per_example"][k]
            c = ci([boot_mean(vals, idx) for idx in draws])
            if c:
                v["ci"][k] = [round(c[0], 4), round(c[1], 4)]
            if BASE in rows and arm != BASE:
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
        print(f"  {arm:26s} consistency {v['unieval_con']}"
              + (f"   differs from {BASE} on: {', '.join(sig)}" if sig else "   (no dimension differs)"))

    for v in rows.values():
        v.pop("per_example", None)
    with open(a.output, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
