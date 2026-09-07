"""Pairwise significance testing over evaluate_summarisation.py result files.

The summarisation counterpart to mcnemar.py. McNemar's test doesn't apply
here: ROUGE is a continuous per-example score, not a right/wrong bit, so there
are no discordant pairs to count. But the pairing is just as real - every model
is scored on the same test articles - and ignoring it would overstate the
uncertainty in exactly the same way.

The test used is a paired bootstrap over the per-example score differences:
resample the test set with replacement, recompute the mean difference on each
resample, and report how often the sign flips. That gives a two-sided p-value
and a confidence interval on the difference, without assuming the per-example
ROUGE scores are normal (they are not - they are bounded in [0, 1] and heavily
skewed).

Usage:
    uv run --project ../evaluate python bootstrap_rouge.py <results-dir-or-files...>
    uv run --project ../evaluate python bootstrap_rouge.py results/ --metric rouge2
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def load_run(path, metric):
    """Read one evaluate_summarisation.py output into per-example scores."""
    with open(path) as fh:
        record = json.load(fh)

    if "per_example_rouge" not in record:
        raise ValueError(
            f"{path}: missing 'per_example_rouge' - not an "
            "evaluate_summarisation.py result file?"
        )

    scores = np.array([row[metric] for row in record["per_example_rouge"]])
    metrics = record.get("metrics", {})
    return {
        "name": Path(path).stem,
        "scores": scores,
        "n": len(scores),
        "mean": float(scores.mean()),
        "pred_words": metrics.get("mean_pred_words", float("nan")),
        "pred_sentences": metrics.get("mean_pred_sentences", float("nan")),
        "empty": metrics.get("num_empty", 0),
    }


def paired_bootstrap(a, b, resamples, rng):
    """Paired bootstrap on the per-example difference a - b."""
    if a["n"] != b["n"]:
        raise ValueError(
            f"{a['name']} has {a['n']} examples but {b['name']} has {b['n']} - "
            "not the same test set, so the results aren't paired"
        )

    differences = a["scores"] - b["scores"]
    observed = float(differences.mean())

    # One index draw per resample, applied to the *differences* - that is what
    # makes this paired: both models are always scored on the same resampled
    # articles.
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[indices].mean(axis=1)

    # Two-sided p-value by the sign-flip convention: how often does a resample
    # land on the other side of zero from the observed difference? The +1s are
    # the standard finite-sample correction, so p is never exactly 0.
    if observed >= 0:
        tail = int((means <= 0).sum())
    else:
        tail = int((means >= 0).sum())
    p = min(1.0, 2 * (tail + 1) / (resamples + 1))

    low, high = np.percentile(means, [2.5, 97.5])
    return {"observed": observed, "p": p, "ci": (float(low), float(high))}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths", nargs="+", help="Result JSON files, or directories containing them."
    )
    parser.add_argument(
        "--metric",
        default="rouge1",
        choices=["rouge1", "rouge2", "rougeL"],
        help="Which ROUGE variant to test on.",
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance threshold."
    )
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

    runs = [load_run(f, args.metric) for f in files]
    runs.sort(key=lambda r: -r["mean"])

    print(
        f"{'model':<34} {'n':>6} {args.metric:>9} {'words':>7} {'sents':>7} {'empty':>6}"
    )
    print("-" * 74)
    for run in runs:
        print(
            f"{run['name']:<34} {run['n']:>6} {run['mean']:>9.4f} "
            f"{run['pred_words']:>7.1f} {run['pred_sentences']:>7.2f} {run['empty']:>6}"
        )

    print()
    print(f"Pairwise paired bootstrap on {args.metric}, {args.resamples} resamples.")
    print("'diff' is mean(A) - mean(B) per example; CI is the 95% interval on it.")
    print()
    print(f"{'A':<26} {'B':<26} {'diff':>8} {'95% CI':>18} {'p':>8}  verdict")
    print("-" * 100)

    rng = np.random.default_rng(args.seed)
    for a, b in itertools.combinations(runs, 2):
        result = paired_bootstrap(a, b, args.resamples, rng)
        if result["p"] < args.alpha:
            better = a["name"] if result["observed"] > 0 else b["name"]
            verdict = f"significant ({better} better)"
        else:
            verdict = "not significant"
        ci = f"[{result['ci'][0]:+.4f},{result['ci'][1]:+.4f}]"
        print(
            f"{a['name']:<26} {b['name']:<26} {result['observed']:>+8.4f} "
            f"{ci:>18} {result['p']:>8.2} {verdict}"
        )

    print()
    print("Caveats: these p-values test one pair at a time - with many models")
    print("compared, apply a multiple-comparison correction before treating any")
    print("single result as established. And ROUGE is a proxy: a significant")
    print("difference of a few tenths of a point is not necessarily a difference")
    print("a reader would notice. Read the generations in the results JSON too.")


if __name__ == "__main__":
    main()
