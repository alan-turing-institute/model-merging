"""Count degenerate generations in results JSONs.

ROUGE cannot tell you WHY a model scored badly. On XSum the distilled students
lost 0.06 ROUGE-1 to the merge they were distilled from, and reading the
generations showed the reason was not worse summaries: it was that roughly one
output in ten had collapsed - an empty string, a lone full stop, or a single
token repeated to the generation cap ("Bes Bes Bes ..." for all 64 tokens).
The surviving summaries were frequently fine and occasionally better than the
merge's. An empty output scores a flat zero, so a tenth of the test set failing
this way accounts for most of the gap.

That distinction matters for what you do next. "Worse at summarising" points at
the teacher signal; "cannot reliably stop" points at the loss balance, and it
was the second. So this is worth measuring alongside the metric rather than
discovering by eye.

Three signals, because each catches a different failure:

  empty     nothing but whitespace or punctuation - scores zero on any metric
  word_run  the same word three or more times consecutively - a decode loop
  trigram   some three-word sequence appearing three or more times - a longer
            loop that word_run misses
  low_uniq  unique-word ratio below 0.5 - degeneracy too diffuse for either

Usage:
    uv run --project ../evaluate python degeneracy.py <results-dir-or-files...>
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def longest_run(words):
    best = cur = 1 if words else 0
    for a, b in zip(words, words[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def analyse(predictions):
    counts = {"empty": 0, "word_run": 0, "trigram": 0, "low_uniq": 0}
    worst = ("", 0)
    for prediction in predictions:
        words = re.findall(r"\w+", (prediction or "").lower())
        if not words:
            counts["empty"] += 1
            continue
        run = longest_run(words)
        if run >= 3:
            counts["word_run"] += 1
        if run > worst[1]:
            worst = (prediction, run)
        trigrams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
        if trigrams and max(trigrams.values()) >= 3:
            counts["trigram"] += 1
        if len(set(words)) / len(words) < 0.5:
            counts["low_uniq"] += 1
    return counts, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--show-worst", action="store_true",
                        help="print the most repetitive generation per model")
    args = parser.parse_args()

    files = []
    for path in args.paths:
        p = Path(path)
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])

    rows = []
    for path in files:
        record = json.loads(path.read_text())
        predictions = record.get("predictions")
        if predictions is None:
            continue  # classification results carry no free text
        counts, worst = analyse(predictions)
        rows.append((path.stem, len(predictions), counts, worst))

    if not rows:
        sys.exit("No results files with a 'predictions' field found.")

    print(f"{'model':<30}{'n':>6}{'empty':>7}{'word_run':>10}{'trigram':>9}{'low_uniq':>10}")
    print("-" * 72)
    for name, n, c, _ in rows:
        print(f"{name:<30}{n:>6}{c['empty']:>7}{c['word_run']:>10}"
              f"{c['trigram']:>9}{c['low_uniq']:>10}")

    if args.show_worst:
        print()
        for name, _, _, worst in rows:
            if worst[1] >= 3:
                print(f"{name} (run of {worst[1]}): {worst[0][:120]}")


if __name__ == "__main__":
    main()
