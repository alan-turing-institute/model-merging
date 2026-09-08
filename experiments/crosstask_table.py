"""Joint table over the cross-task merging results.

run_crosstask_pipeline.sh evaluates every model on BOTH test sets and names the
outputs <stem>.crime.json and <stem>.xsum.json. This pairs them back up.

The headline metrics alone don't answer the question. "F1 0.91 on crime and
ROUGE-1 0.31 on XSum" means nothing until you know what the base model scored
and what each single-task expert scored. So the table reports RETENTION per
task, on the scale that matters:

    retention = (merged - base) / (expert - base)

    1.0   keeps everything fine-tuning bought on that task
    0.0   back to base behaviour - the merge dropped that skill entirely
    < 0   worse than never having fine-tuned at all: active interference

A merge that scores 0.9 on both is the result the whole method promises. One
that scores ~1.0 on one task and ~0.0 on the other has not merged anything, it
has picked a winner. Two middling numbers mean the deltas are fighting.

Also printed, and worth reading before the retention columns: `unparsed` (crime
generations that were not one of the two labels) and `words` (mean XSum output
length). Format leakage between the two tasks shows up there first - a
classifier that starts writing sentences, or a summariser that starts emitting
"not_crime".

Usage:
    uv run --project ../evaluate python crosstask_table.py <results-dir>
"""

import argparse
import json
import sys
from pathlib import Path

CRIME_SUFFIX = ".crime.json"
XSUM_SUFFIX = ".xsum.json"


def load_results(directory):
    """Collect {stem: {"crime": metrics, "xsum": metrics}} from a directory."""
    runs = {}
    for path in sorted(Path(directory).glob("*.json")):
        name = path.name
        if name.endswith(CRIME_SUFFIX):
            stem, task = name[: -len(CRIME_SUFFIX)], "crime"
        elif name.endswith(XSUM_SUFFIX):
            stem, task = name[: -len(XSUM_SUFFIX)], "xsum"
        else:
            continue
        record = json.loads(path.read_text())
        runs.setdefault(stem, {})[task] = record.get("metrics", {})
        runs[stem].setdefault("provenance", record.get("provenance"))
    return runs


def retention(merged, base, expert):
    """Fraction of the expert's gain over base that `merged` keeps."""
    if merged is None or base is None or expert is None:
        return None
    span = expert - base
    # An expert that never beat base on its own task gives no scale to measure
    # against; reporting a ratio there would be noise dressed as a number.
    if abs(span) < 1e-9:
        return None
    return (merged - base) / span


def fmt(value, spec=".3f", width=8):
    return f"{'-':>{width}}" if value is None else f"{value:>{width}{spec}}"


NAME_WIDTH = 26


def display(stem):
    """Shorten a registry-style stem to something that fits the name column."""
    short = stem.replace("gemma3-crosstask-merged-", "merged-")
    if len(short) > NAME_WIDTH:
        short = short[: NAME_WIDTH - 1] + "\u2026"
    return short


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results_dir", help="Directory of *.crime.json / *.xsum.json")
    parser.add_argument("--base-stem", default="base")
    parser.add_argument("--crime-expert-stem", default="crime-expert")
    parser.add_argument("--xsum-expert-stem", default="xsum-expert")
    args = parser.parse_args()

    runs = load_results(args.results_dir)
    if not runs:
        sys.exit(f"No *.crime.json / *.xsum.json files found in {args.results_dir}")

    def metric(stem, task, key):
        return runs.get(stem, {}).get(task, {}).get(key)

    base_f1 = metric(args.base_stem, "crime", "f1")
    base_rouge = metric(args.base_stem, "xsum", "rouge1")
    expert_f1 = metric(args.crime_expert_stem, "crime", "f1")
    expert_rouge = metric(args.xsum_expert_stem, "xsum", "rouge1")

    missing = [
        label
        for label, value in (
            (f"{args.base_stem}.crime", base_f1),
            (f"{args.base_stem}.xsum", base_rouge),
            (f"{args.crime_expert_stem}.crime", expert_f1),
            (f"{args.xsum_expert_stem}.xsum", expert_rouge),
        )
        if value is None
    ]
    if missing:
        print(
            "WARNING: retention needs the base and both experts; missing "
            + ", ".join(missing)
            + "\n         retention columns will be blank.\n"
        )

    header = (
        f"{'':<{NAME_WIDTH}} | {'acc':>7} {'F1':>7} {'unpars':>6} {'ret':>4} "
        f"| {'R-1':>7} {'R-2':>7} {'words':>6} {'ret':>4}"
    )
    print(f"{'model':<{NAME_WIDTH}} | {'CRIME':^27} | {'XSUM':^27}")
    print(header)
    print("-" * len(header))

    # Reference rows first, in the order they should be read, then the merges.
    reference = [args.base_stem, args.crime_expert_stem, args.xsum_expert_stem]
    ordered = [s for s in reference if s in runs] + sorted(
        s for s in runs if s not in reference
    )

    for stem in ordered:
        crime_f1 = metric(stem, "crime", "f1")
        rouge1 = metric(stem, "xsum", "rouge1")
        print(
            f"{display(stem):<{NAME_WIDTH}} | "
            f"{fmt(metric(stem, 'crime', 'accuracy'), '.3f', 7)} "
            f"{fmt(crime_f1, '.3f', 7)} "
            f"{fmt(metric(stem, 'crime', 'num_unparsed'), 'd', 6)} "
            f"{fmt(retention(crime_f1, base_f1, expert_f1), '.2f', 4)} | "
            f"{fmt(rouge1, '.3f', 7)} "
            f"{fmt(metric(stem, 'xsum', 'rouge2'), '.3f', 7)} "
            f"{fmt(metric(stem, 'xsum', 'mean_pred_words'), '.1f', 6)} "
            f"{fmt(retention(rouge1, base_rouge, expert_rouge), '.2f', 4)}"
        )

    print()
    print("ret = (merged - base) / (expert - base) on that task: 1.0 keeps everything")
    print("fine-tuning bought, 0.0 is back to base, negative is active interference.")
    print()
    print("Read 'unpars' and 'words' before the retention columns. Format leakage")
    print("between the two tasks shows up there first - a classifier that starts")
    print("writing sentences, or a summariser emitting \"not_crime\".")
    print()
    print("Caveat: single runs, no seed variation. The crime arm's two half-data")
    print("models differed by 3.7pp from training noise alone, so treat gaps")
    print("smaller than that as unresolved rather than as findings.")


if __name__ == "__main__":
    main()
