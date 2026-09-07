"""Pairwise significance testing over evaluate.py result files.

evaluate.py saves per-example `predictions` and `actuals` alongside its
metrics, which means differences between two models can be tested properly
rather than eyeballed from accuracy alone. Two models evaluated on the same
test set give *paired* observations, so the right test is McNemar's: it looks
only at the examples where the two models disagree, and asks whether the
disagreements are lopsided beyond chance.

Comparing raw accuracies would ignore that pairing and badly overstate the
uncertainty - two models can differ by 1pp while disagreeing on only a handful
of examples (significant) or on hundreds (not).

The exact form is used (a binomial test on the discordant pairs) rather than
the chi-square approximation, since the discordant counts here are small
enough that the approximation is unreliable.

Usage:
    uv run --project ../evaluate python mcnemar.py <results-dir-or-files...>
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

from scipy.stats import binomtest


def load_run(path):
    """Read one evaluate.py output into (name, per-example correctness)."""
    with open(path) as fh:
        record = json.load(fh)

    for key in ("predictions", "actuals"):
        if key not in record:
            raise ValueError(f"{path}: missing '{key}' - not an evaluate.py result file?")

    predictions = record["predictions"]
    actuals = record["actuals"]
    if len(predictions) != len(actuals):
        raise ValueError(f"{path}: {len(predictions)} predictions vs {len(actuals)} actuals")

    # An unparsed generation (None) is scored wrong, matching evaluate.py.
    correct = [p is not None and p == a for p, a in zip(predictions, actuals)]
    return {
        "name": Path(path).stem,
        "correct": correct,
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "unparsed": sum(p is None for p in predictions),
    }


def mcnemar(a, b):
    """Exact McNemar's test between two runs' correctness vectors."""
    if a["n"] != b["n"]:
        raise ValueError(
            f"{a['name']} has {a['n']} examples but {b['name']} has {b['n']} - "
            "not the same test set, so the results aren't paired"
        )

    # Discordant pairs: examples where exactly one of the two got it right.
    a_only = sum(1 for x, y in zip(a["correct"], b["correct"]) if x and not y)
    b_only = sum(1 for x, y in zip(a["correct"], b["correct"]) if y and not x)
    discordant = a_only + b_only

    if discordant == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p": 1.0}

    # Under H0 each discordant example is a coin flip as to which model wins.
    p = binomtest(a_only, discordant, 0.5).pvalue
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="Result JSON files, or directories containing them.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold for the verdict column.")
    args = parser.parse_args()

    files = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)

    if len(files) < 2:
        sys.exit(f"Need at least 2 result files to compare, found {len(files)}")

    runs = [load_run(f) for f in files]

    print(f"{'model':<34} {'n':>6} {'accuracy':>9} {'unparsed':>9}")
    print("-" * 62)
    for run in sorted(runs, key=lambda r: -r["accuracy"]):
        print(f"{run['name']:<34} {run['n']:>6} {run['accuracy']:>9.4f} {run['unparsed']:>9}")

    print()
    print("Pairwise McNemar (exact). 'only' columns count examples that model got")
    print("right and the other got wrong; p tests whether that split is lopsided.")
    print()
    print(f"{'A':<26} {'B':<26} {'A only':>7} {'B only':>7} {'p':>9}  verdict")
    print("-" * 92)

    for a, b in itertools.combinations(sorted(runs, key=lambda r: -r["accuracy"]), 2):
        result = mcnemar(a, b)
        if result["discordant"] == 0:
            verdict = "identical predictions"
        elif result["p"] < args.alpha:
            better = a["name"] if result["a_only"] > result["b_only"] else b["name"]
            verdict = f"significant ({better} better)"
        else:
            verdict = "not significant"
        print(
            f"{a['name']:<26} {b['name']:<26} "
            f"{result['a_only']:>7} {result['b_only']:>7} {result['p']:>9.2}  {verdict}"
        )

    print()
    print("Caveat: these p-values test one pair at a time. With many models compared,")
    print("apply a multiple-comparison correction (e.g. Bonferroni: divide alpha by")
    print("the number of pairs) before treating any single result as established.")


if __name__ == "__main__":
    main()
