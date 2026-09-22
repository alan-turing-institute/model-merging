"""Compare generations across runs, focusing on the examples that failed before.

`degeneracy.py` answers "did the outputs collapse?" and after the loss-weighting
fix the answer is a clean zero on all four of its signals. That is necessary but
not sufficient: zero collapses is compatible with summaries that are bland,
truncated mid-sentence, or identical to each other. The counters were built to
catch a specific pathology and they cannot see anything else.

So this script asks the narrower question the fix actually has to answer: the
examples the broken run got wrong - what does the repaired run produce for those
same examples? The test set is fixed and ordered, so index i is the same article
in every results file, and the comparison can be made per example rather than
per mean.

Two outputs:

  a table of quality signals the degeneracy counters do not cover, including
  mean ROUGE restricted to the anchor run's failed indices, which is the number
  that says whether the repair reached the failures or merely avoided them;

  the failing cases printed side by side, because a table cannot tell you
  whether a summary is *right* and at some point someone has to read them.

Usage:
    uv run --project ../evaluate python compare_outputs.py \
        --anchor broken.json fixed.json merge.json full.json --show 10
"""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def words_of(text):
    return re.findall(r"\w+", (text or "").lower())


def longest_run(words):
    best = cur = 1 if words else 0
    for a, b in zip(words, words[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def is_degenerate(prediction):
    """The same four signals degeneracy.py uses, collapsed to one boolean."""
    words = words_of(prediction)
    if not words:
        return True
    if longest_run(words) >= 3:
        return True
    trigrams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    if trigrams and max(trigrams.values()) >= 3:
        return True
    return len(set(words)) / len(words) < 0.5


def load(path):
    record = json.loads(Path(path).read_text())
    if record.get("predictions") is None:
        raise SystemExit(f"{path} has no 'predictions' field (classification result?)")
    rouge = [row["rouge1"] for row in record.get("per_example_rouge", [])]
    return {
        "name": Path(path).stem,
        "predictions": record["predictions"],
        "references": record["references"],
        "rouge1": rouge,
    }


def profile(predictions, rouge1, failed_idx):
    """Signals the collapse counters miss."""
    lengths = [len(p.split()) for p in predictions]
    # A summary that stops mid-clause is not degenerate by any of the four
    # signals - it has no repetition and plenty of unique words - but it is
    # still a failure, and it is the failure mode a token budget produces.
    unterminated = sum(1 for p in predictions if p.strip() and p.strip()[-1] not in ".!?")
    # Distinct articles receiving the identical summary is a collapse the
    # per-example signals cannot see, because each output on its own is fine.
    duplicates = len(predictions) - len({p.strip() for p in predictions})
    uniq = [len(set(w)) / len(w) for w in map(words_of, predictions) if w]
    return {
        "n": len(predictions),
        "empty": sum(1 for p in predictions if not p.strip()),
        "words_mean": statistics.fmean(lengths) if lengths else 0.0,
        "words_p10": sorted(lengths)[len(lengths) // 10] if lengths else 0,
        "words_p90": sorted(lengths)[len(lengths) * 9 // 10] if lengths else 0,
        "unterminated%": 100 * unterminated / len(predictions),
        "duplicate%": 100 * duplicates / len(predictions),
        "uniq_ratio": statistics.fmean(uniq) if uniq else 0.0,
        "rouge1": statistics.fmean(rouge1) if rouge1 else float("nan"),
        "rouge1_failed": (
            statistics.fmean([rouge1[i] for i in failed_idx])
            if rouge1 and failed_idx else float("nan")
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--anchor", default=None,
                        help="results file whose degenerate outputs define the "
                             "examples of interest (default: the first file)")
    parser.add_argument("--show", type=int, default=8,
                        help="how many failing cases to print side by side")
    args = parser.parse_args()

    runs = [load(path) for path in args.files]
    anchor = next((r for r in runs if r["name"] == Path(args.anchor).stem), runs[0]) \
        if args.anchor else runs[0]

    failed_idx = [i for i, p in enumerate(anchor["predictions"]) if is_degenerate(p)]
    print(f"anchor: {anchor['name']} - {len(failed_idx)} degenerate of "
          f"{len(anchor['predictions'])} "
          f"({100 * len(failed_idx) / len(anchor['predictions']):.1f}%)\n")

    header = (f"{'model':<26}{'n':>5}{'empty':>6}{'words':>7}{'p10':>5}{'p90':>5}"
              f"{'unterm%':>9}{'dup%':>6}{'uniq':>6}{'ROUGE1':>8}{'on-failed':>11}")
    print(header)
    print("-" * len(header))
    for run in runs:
        p = profile(run["predictions"], run["rouge1"], failed_idx)
        print(f"{run['name']:<26}{p['n']:>5}{p['empty']:>6}{p['words_mean']:>7.1f}"
              f"{p['words_p10']:>5}{p['words_p90']:>5}{p['unterminated%']:>9.1f}"
              f"{p['duplicate%']:>6.1f}{p['uniq_ratio']:>6.2f}"
              f"{p['rouge1']:>8.4f}{p['rouge1_failed']:>11.4f}")

    if not failed_idx:
        print("\nNo degenerate outputs in the anchor run - nothing to read.")
        return

    print(f"\n{'=' * 78}\nThe previously-failing examples\n{'=' * 78}")
    for i in failed_idx[:args.show]:
        print(f"\n--- test index {i} ---")
        print(f"REFERENCE : {anchor['references'][i]}")
        for run in runs:
            score = f" [{run['rouge1'][i]:.3f}]" if run["rouge1"] else ""
            print(f"{run['name'][:10]:<10}{score}: {run['predictions'][i]!r}")


if __name__ == "__main__":
    main()
