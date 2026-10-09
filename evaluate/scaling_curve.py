"""Read a data-scaling curve's .eval logs and say whether the curve is flat.

    uv run python scaling_curve.py logs --task samsum
    uv run python scaling_curve.py logs --task billsum --metric rougeLsum

WHAT THIS IS FOR. A scaling curve exists to answer one question before a task is
given forty arms the way XSum was: does the score still improve as training data
is added? If it does, the task has headroom and differences between merge methods
have somewhere to show up. If it does not, the task is saturated and no merging
experiment on it can work - which is worth knowing after two GPU-hours rather
than after a month.

HOW IT DECIDES. Not by eye on a column of means. Consecutive arms are compared by
a PAIRED BOOTSTRAP over test examples - the same resampling used for the XSum
re-scoring and the TIES-vs-linear seeds - because the arms are scored on
identical examples, so pairing removes the example-to-example variance that
otherwise swamps a difference this size. A flat step is one whose confidence
interval contains zero.

WHAT IT CANNOT DO is tell you that a non-flat step is a real effect of data
rather than of the training draw: there is one run per arm, so the seed-to-seed
spread is unmeasured. On the crime task that spread was 0.004-0.005, larger than
the whole spread between seven merge methods. Read a step smaller than that as
flat whatever its interval says, and if a decision turns on it, run the arm again
at a second seed.

WHY THE MERGING READ IS DIFFERENT FROM THE SATURATION READ. A flat top step says
the task is saturated. But the quantity that actually constrains a merging
experiment is not the top of the curve, it is the step from the arm each EXPERT
would be trained on to the arm the FULL-DATA model is trained on - because the
recovery fraction R = (merged - best expert) / (full - best expert) has exactly
that step as its denominator. So this script prints, for every arm, the step
from half that arm's rows to all of them: read the row for the full-data size
you intend to use, and that step is the denominator you would be dividing by.
A two-way split at 1,000 rows and one at 8,000 rows are completely different
experiments on the same dataset, and the curve is what tells them apart.

This supersedes billsum_curve.py, which is the same logic hardcoded to BillSum's
four fraction-named arms.
"""

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

REPORTED = ["rouge1", "rouge2", "rougeL", "rougeLsum", "semantic",
            "entity_support", "pred_words", "ref_words", "unterminated"]

# Per task: the headline metric, and how an arm name maps to a training-row
# count. BillSum named its arms by fraction of a --train-size; SAMSum names them
# by row count directly, which is why this is a table rather than one rule.
TASKS = {
    # rougeLsum for BillSum: multi-sentence references, where rougeL's
    # whole-string LCS punishes covering the same provisions in a different
    # order.
    "billsum": {"headline": "rougeLsum", "default_cap": 640,
                "fractions": {"base": 0.0, "eighth": 0.125, "quarter": 0.25,
                              "half": 0.5, "full": 1.0}},
    # rouge1 for SAMSum: references are one or two sentences, so rougeLsum and
    # rougeL are nearly identical and neither adds anything; rouge1 is what the
    # SAMSum literature quotes.
    "samsum": {"headline": "rouge1", "default_cap": 128, "fractions": None},
}

BOOTSTRAP_RESAMPLES = 2000

# Expert-draw standard deviation measured on the crime task across five seeds.
# A gap smaller than this cannot be attributed to the method rather than the draw.
SEED_SPREAD = 0.005


def arm_of(log, task: str) -> str | None:
    """Which curve arm a log belongs to, from the adapter or model name."""
    model_args = log.eval.model_args or {}
    text = str(model_args.get("adapter_path") or log.eval.model)
    match = re.search(rf"{task}-([A-Za-z0-9]+)-lora", text)
    if match:
        return match.group(1)
    # No adapter means the untrained base model: the n=0 point.
    if "adapter_path" not in model_args:
        return "base"
    return None


def rows_of(arm: str, task: str, train_size: int) -> int:
    """Training rows for an arm, for the x-axis."""
    if arm == "base":
        return 0
    fractions = TASKS[task]["fractions"]
    if fractions is not None:
        return int(fractions[arm] * train_size)
    match = re.fullmatch(r"n(\d+)", arm)
    if not match:
        raise SystemExit(f"cannot read a row count from arm name {arm!r}")
    return int(match.group(1))


def per_sample(log) -> dict[str, dict]:
    """{metric: {sample_id: value}} for one log.

    Keyed by sample id, not position, so two arms are paired on the same example
    even if Inspect wrote their samples in a different order.
    """
    out: dict[str, dict] = defaultdict(dict)
    for sample in log.samples or []:
        for scorer_scores in (sample.scores or {}).values():
            value = scorer_scores.value
            if not isinstance(value, dict):
                continue
            for metric, v in value.items():
                if isinstance(v, (int, float)) and v == v:   # drop NaN (ungradeable)
                    out[metric][sample.id] = float(v)
    return out


def paired_bootstrap(a: dict, b: dict, resamples: int = BOOTSTRAP_RESAMPLES):
    """95% CI for mean(b) - mean(a) over the examples both arms scored.

    Deterministic seed so a reported interval can be reproduced exactly.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diffs = [b[i] - a[i] for i in shared]
    point = statistics.fmean(diffs)
    rng = random.Random(0)
    n = len(diffs)
    means = sorted(
        statistics.fmean([diffs[rng.randrange(n)] for _ in range(n)])
        for _ in range(resamples)
    )
    return point, means[int(0.025 * resamples)], means[int(0.975 * resamples)], n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", nargs="?", default="logs", type=Path)
    parser.add_argument("--task", default="samsum", choices=sorted(TASKS))
    parser.add_argument("--metric", default=None, help="defaults to the task's headline")
    parser.add_argument("--train-size", type=int, default=4000,
                        help="rows in the full arm; only used for fraction-named arms")
    parser.add_argument("--json", type=Path,
                        help="also write the curve here, to commit alongside the .eval logs")
    args = parser.parse_args()
    spec = TASKS[args.task]
    metric = args.metric or spec["headline"]

    logs = {}
    for info in list_eval_logs(str(args.log_dir)):
        log = read_eval_log(info)
        if log.eval.task.split("/")[-1] != args.task or log.status != "success":
            continue
        arm = arm_of(log, args.task)
        if arm is None:
            continue
        # Keep the most recent log per arm, so a re-run supersedes rather than
        # silently competing with what it replaced.
        if arm not in logs or log.eval.created > logs[arm].eval.created:
            logs[arm] = log

    if not logs:
        raise SystemExit(f"no {args.task} logs in {args.log_dir}")

    # EVERY POINT ON THE CURVE MUST SHARE ONE GENERATION CAP. The cap truncates
    # long outputs, so an arm scored at 128 tokens and one scored at 256 are not
    # measuring the same thing, and the difference would be read as an effect of
    # training data. On BillSum the base model left 24% of its generations
    # unterminated at 640 and the smallest arm 10%, so the whole curve had to be
    # re-scored. Refuse to compare across caps rather than quietly averaging.
    caps = {arm: (log.eval.task_args or {}).get("max_tokens", spec["default_cap"])
            for arm, log in logs.items()}
    if len(set(caps.values())) > 1:
        detail = ", ".join(f"{arm}={cap}" for arm, cap in sorted(caps.items()))
        raise SystemExit(
            f"arms were scored under different generation caps ({detail}).\n"
            f"Re-score the odd ones out with -T max_tokens=<the common value> "
            f"before reading this curve."
        )

    present = sorted(logs, key=lambda a: rows_of(a, args.task, args.train_size))
    samples = {arm: per_sample(logs[arm]) for arm in present}
    rows = {arm: rows_of(arm, args.task, args.train_size) for arm in present}

    width = max(len(m) for m in REPORTED) + 2
    print(f"{'rows':>6}  {'arm':<8}  " + "".join(f"{m:>{width}}" for m in REPORTED))
    for arm in present:
        cells = []
        for m in REPORTED:
            values = samples[arm].get(m)
            cells.append(f"{statistics.fmean(values.values()):>{width}.4f}"
                         if values else f"{'-':>{width}}")
        print(f"{rows[arm]:>6}  {arm:<8}  " + "".join(cells))

    for arm in present:
        unterminated = samples[arm].get("unterminated", {})
        if unterminated and statistics.fmean(unterminated.values()) > 0.1:
            share = statistics.fmean(unterminated.values())
            print(f"\nWARNING: {arm} left {share:.0%} of generations unterminated - "
                  f"raise -T max_tokens before reading this curve.")

    def step(lower, upper):
        return paired_bootstrap(samples[lower].get(metric, {}), samples[upper].get(metric, {}))

    print(f"\npaired bootstrap on {metric}, consecutive arms "
          f"({BOOTSTRAP_RESAMPLES} resamples):")
    for lower, upper in zip(present, present[1:]):
        result = step(lower, upper)
        if result is None:
            print(f"  {lower:>8} -> {upper:<8}  no shared examples")
            continue
        point, lo, hi, n = result
        verdict = "flat" if lo <= 0 <= hi else (
            "rising, but below seed spread" if abs(point) < SEED_SPREAD else "rising")
        print(f"  {lower:>8} -> {upper:<8}  {point:+.4f}  "
              f"CI [{lo:+.4f}, {hi:+.4f}]  n={n}  {verdict}")

    # THE MERGING READ. For each arm, the step from half its rows to all of them
    # is the denominator of R if that arm is the full-data model and a two-way
    # split is the experiment. Printed separately because it is a different
    # question from saturation and the answer can differ: a curve can be flat at
    # the top and still have a usable denominator lower down.
    print(f"\ndenominator of R for a two-way split, by full-data size "
          f"(step from n/2 to n on {metric}):")
    by_rows = {rows[a]: a for a in present}
    any_usable = False
    for arm in present:
        half = by_rows.get(rows[arm] // 2)
        if half is None or rows[arm] == 0:
            continue
        result = step(half, arm)
        if result is None:
            continue
        point, lo, hi, _ = result
        usable = point >= 10 * SEED_SPREAD
        any_usable = any_usable or usable
        print(f"  full={rows[arm]:<6} experts={rows[arm] // 2:<6}  "
              f"denominator {point:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'USABLE' if usable else 'too small'}"
              f"  ({point / SEED_SPREAD:.1f}x seed spread)")
    if not any_usable:
        print(f"  none reach 10x the seed spread ({10 * SEED_SPREAD:.3f}). A merging "
              f"experiment at any of these sizes would be estimating R as a ratio "
              f"of two noise-sized numbers, which is what happened on XSum.")

    if args.json:
        payload = {
            "task": args.task,
            "metric": metric,
            "max_tokens": next(iter(set(caps.values()))),
            "test_examples": len(next(iter(samples.values())).get(metric, {})),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed_spread_assumed": SEED_SPREAD,
            "note": "one run per arm; seed spread unmeasured here. On the crime task "
                    "it was 0.004-0.005, so read any step below that as flat.",
            "arms": [
                {"arm": arm, "rows": rows[arm],
                 **{m: round(statistics.fmean(samples[arm][m].values()), 4)
                    for m in REPORTED if samples[arm].get(m)}}
                for arm in present
            ],
            "steps": [
                {"from": lower, "to": upper, "delta": round(r[0], 4),
                 "ci_low": round(r[1], 4), "ci_high": round(r[2], 4), "n": r[3],
                 "flat": r[1] <= 0 <= r[2]}
                for lower, upper in zip(present, present[1:])
                if (r := step(lower, upper)) is not None
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
