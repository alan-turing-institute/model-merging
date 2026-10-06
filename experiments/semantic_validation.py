#!/usr/bin/env python3
"""Does the semantic metric measure what we use it to measure?

The metric is cosine similarity between Sentence-BERT embeddings of a generated
summary and its reference. We use it as a check on whether a ROUGE difference
reflects meaning or only wording. That use rests on two assumptions, and
semantic_metrics.py's own docstring doubts the second:

  1. a correct summary phrased differently scores HIGH  (the reason it exists)
  2. a summary that is wrong scores LOW                 (never tested)

This tests both by perturbing the REFERENCE summary in ways whose ground truth
is known, and reading semantic, ROUGE-1 and entity_support on each. A metric fit
for purpose should keep paraphrase high and push every meaning-breaking
perturbation low. The anecdote in the docstring - Mick Lally becoming Sean Lally
at 73 - predicts that entity and number swaps stay high, which would be a
quantified blind spot rather than an impression.

CPU only: MiniLM over a few thousand short sentences is seconds.
"""
import argparse, json, random, re, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluate"))
from semantic_metrics import embedder, entities, unsupported_entities  # noqa: E402
from rouge_score import rouge_scorer  # noqa: E402

AUX = (" is ", " was ", " are ", " were ", " has ", " have ", " had ", " will ", " can ")


def swap_entity(text, donors, rng):
    """Replace the first detected entity with one from another article."""
    found = entities(text)
    if not found or not donors:
        return None
    return text.replace(found[0], rng.choice(donors), 1)


def swap_number(text, rng):
    nums = re.findall(r"\b\d+\b", text)
    if not nums:
        return None
    n = rng.choice(nums)
    return text.replace(n, str(int(n) + rng.choice([7, 9, 11, 13])), 1)


def negate(text):
    for a in AUX:
        if a in text:
            return text.replace(a, a[:-1] + " not ", 1)
    return None


def shuffle(text, rng):
    w = text.split()
    if len(w) < 4:
        return None
    rng.shuffle(w)
    return " ".join(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="a results JSON carrying predictions/references")
    ap.add_argument("--second", default=None, help="a second arm's results, for natural paraphrase pairs")
    ap.add_argument("--dataset", required=True, help="xsum test split, for the source documents")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    d = json.load(open(args.results))
    refs, preds = d["references"][: args.limit], d["predictions"][: args.limit]
    preds2 = json.load(open(args.second))["predictions"][: args.limit] if args.second else None

    from datasets import load_from_disk
    ds = load_from_disk(args.dataset)
    col = "document" if "document" in ds.column_names else ds.column_names[0]
    docs = [ds[i][col] for i in range(min(args.limit, len(ds)))]

    model = embedder()
    if model is None:
        raise SystemExit("no embedder available - this experiment is meaningless without it")
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    donors = [e for r in refs for e in entities(r)]

    def sim(a, b):
        v = model.encode([a, b], normalize_embeddings=True)
        return float(v[0] @ v[1])

    # (name, candidate builder, ground truth: does it preserve meaning?)
    CONDITIONS = [
        ("identity",        lambda i: refs[i],                          True),
        ("model_output",    lambda i: preds[i],                         True),
        ("paraphrase_nat",  lambda i: preds2[i] if preds2 else None,    True),
        ("entity_swap",     lambda i: swap_entity(refs[i], donors, rng), False),
        ("number_swap",     lambda i: swap_number(refs[i], rng),        False),
        ("negation",        lambda i: negate(refs[i]),                  False),
        ("word_shuffle",    lambda i: shuffle(refs[i], rng),            False),
        ("unrelated",       lambda i: refs[(i + 500) % len(refs)],      False),
    ]

    out = {}
    for name, build, preserves in CONDITIONS:
        sem, r1, sup = [], [], []
        for i in range(len(refs)):
            cand = build(i)
            if not cand:
                continue
            sem.append(sim(cand, refs[i]))
            r1.append(scorer.score(refs[i], cand)["rouge1"].fmeasure)
            found = entities(cand)
            if found:
                bad = unsupported_entities(cand, docs[i] if i < len(docs) else "")
                sup.append((len(found) - len(bad)) / len(found))
        if not sem:
            continue
        out[name] = {
            "preserves_meaning": preserves, "n": len(sem),
            "semantic": st.fmean(sem), "semantic_sd": st.pstdev(sem),
            "rouge1": st.fmean(r1),
            "entity_support": st.fmean(sup) if sup else None,
        }
        print(f"{name:16s} n={len(sem):4d}  semantic {st.fmean(sem):.4f}  "
              f"rouge1 {st.fmean(r1):.4f}  "
              f"entity_support {(f'{st.fmean(sup):.4f}' if sup else '  n/a ')}  "
              f"{'meaning preserved' if preserves else 'MEANING BROKEN'}", flush=True)

    ok = [v["semantic"] for k, v in out.items() if v["preserves_meaning"] and k != "identity"]
    bad = [v["semantic"] for v in out.values() if not v["preserves_meaning"]]
    if ok and bad:
        out["separation"] = {
            "meaning_preserved_mean": st.fmean(ok),
            "meaning_broken_mean": st.fmean(bad),
            "gap": st.fmean(ok) - st.fmean(bad),
        }
        print(f"\nseparation: preserved {st.fmean(ok):.4f} vs broken {st.fmean(bad):.4f} "
              f"-> gap {st.fmean(ok) - st.fmean(bad):+.4f}")
        print("A metric fit for this purpose separates these cleanly. Per-condition "
              "values matter more than the mean: a high score on entity_swap or "
              "number_swap is the documented blind spot, quantified.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
