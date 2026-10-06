#!/usr/bin/env python3
"""Score the bake-off's perturbations with COMET, in an isolated environment.

COMET is a learned metric trained on human judgements of machine TRANSLATION:
source, hypothesis and reference are length-matched and meaning-equivalent
across languages. Summarisation hands it a 400-word article and a 20-word
summary that is deliberately not equivalent to its source, which is well outside
its training distribution. A learned metric off-distribution does not degrade
gracefully - it returns confident numbers with no calibration behind them - so
this measures whether it discriminates rather than assuming it does.

It reads the triples the bake-off dumped rather than rebuilding them, so the two
environments agree on a file instead of on an RNG call order.
"""
import argparse, json, statistics as st
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    from comet import load_from_checkpoint
    import glob, torch

    root = snapshot_download(a.model)
    ckpt = glob.glob(str(Path(root) / "checkpoints" / "*.ckpt"))
    if not ckpt:
        raise SystemExit("no .ckpt under " + root)
    model = load_from_checkpoint(ckpt[0])
    gpus = 1 if torch.cuda.is_available() else 0
    print("comet loaded, gpus=%d" % gpus, flush=True)

    conds = json.load(open(a.conditions))
    out = {}
    for name, c in conds.items():
        data = [{"src": it["doc"], "mt": it["cand"], "ref": it["target"]}
                for it in c["items"]]
        if not data:
            continue
        res = model.predict(data, batch_size=a.batch_size, gpus=gpus, progress_bar=False)
        scores = list(res["scores"])
        out[name] = {"n": len(scores), "comet": st.fmean(scores),
                     "says_what_reference_says": c["says_what_reference_says"],
                     "faithful_to_article": c["faithful_to_article"]}
        print("%-16s n=%4d comet=%.4f  %s" % (
            name, len(scores), st.fmean(scores),
            "meaning preserved" if c["says_what_reference_says"] else "MEANING BROKEN"), flush=True)

    yes = [v["comet"] for k, v in out.items() if v["says_what_reference_says"] and k != "identity"]
    no = [v["comet"] for v in out.values() if not v["says_what_reference_says"]]
    if yes and no:
        out["separation"] = {"preserved": st.fmean(yes), "broken": st.fmean(no),
                             "gap": st.fmean(yes) - st.fmean(no)}
        print("\nseparation: preserved %.4f vs broken %.4f -> %+.4f"
              % (st.fmean(yes), st.fmean(no), st.fmean(yes) - st.fmean(no)))
        print("Compare: UniEval consistency +0.562, LLM judge +0.308, entailment +0.266, "
              "BERTScore -0.003, ROUGE-1 -0.249.")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=2)
    print("\nwrote " + a.output)


if __name__ == "__main__":
    main()
