"""Read the BillSum scaling-curve .eval logs and say whether the curve is flat.

    uv run python billsum_curve.py logs

WHAT THIS IS FOR. The curve exists to answer one question before we spend forty
arms on BillSum the way we spent forty on XSum: does the score still improve
between half the training data and all of it? If it does, the task has headroom
and differences between merge methods have somewhere to show up. If it does not,
the task is saturated and no merging experiment on it can work, which is worth
knowing after two GPU-hours rather than after a month.

HOW IT DECIDES. Not by eye on four means. Consecutive arms are compared by a
PAIRED BOOTSTRAP over test examples - the same resampling used for the XSum
re-scoring and the TIES-vs-linear seeds - because the arms are scored on
identical examples, so pairing removes the example-to-example variance that
otherwise swamps a difference this size. A flat step is one whose confidence
interval contains zero.

ONE THING THIS CANNOT DO is tell you that a non-flat step is a real effect of
data rather than of the training draw: there is one run per arm, so the
seed-to-seed spread is unmeasured here. On the crime task that spread was
0.004-0.005, larger than the whole spread between seven merge methods. Read a
step smaller than that as flat whatever its interval says, and if the decision
turns on it, run the arm again at a second seed.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

# Headline first: rougeLsum is the multi-sentence summarisation standard and what
# BillSum papers quote. The rest are read for the shape of the failure, not the
# ranking.
HEADLINE = "rougeLsum"
REPORTED = [HEADLINE, "rouge1", "rouge2", "semantic", "entity_support",
            "pred_words", "ref_words", "unterminated"]

# Arm name -> training rows, for the x-axis. Derived from the arm name rather
# than read from the log because the log records the model, not the dataset.
ARM_ORDER = ["base", "eighth", "quarter", "half", "full"]
FRACTION = {"base": 0.0, "eighth": 0.125, "quarter": 0.25, "half": 0.5, "full": 1.0}

BOOTSTRAP_RESAMPLES = 2000

# Expert-draw standard deviation measured on the crime task across five seeds.
# A gap smaller than this cannot be attributed to the method rather than the draw.
SEED_SPREAD = 0.005

# Matches billsum_task.py's default, for logs written before the cap was a task arg.
DEFAULT_MAX_TOKENS = 640


def arm_of(log) -> str | None:
    """Which curve arm a log belongs to, from the adapter or model name."""
    name = (log.eval.model_args or {}).get("adapter_path") or log.eval.model
    text = str(name)
    for arm in ("eighth", "quarter", "half", "full"):
        if f"billsum-{arm}-lora" in text:
            return arm
    # No adapter on a billsum log means the untrained base model: the n=0 point.
    if "adapter_path" not in (log.eval.model_args or {}):
        return "base"
    return None


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
    import random

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
    lo = means[int(0.025 * resamples)]
    hi = means[int(0.975 * resamples)]
    return point, lo, hi, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", nargs="?", default="logs", type=Path)
    parser.add_argument("--metric", default=HEADLINE, help="metric to test the curve on")
    parser.add_argument("--train-size", type=int, default=4000,
                        help="rows in the full arm, for the x-axis")
    parser.add_argument("--json", type=Path,
                        help="also write the curve here, to commit alongside the .eval logs")
    args = parser.parse_args()

    logs = {}
    for info in list_eval_logs(str(args.log_dir)):
        log = read_eval_log(info)
        if log.eval.task.split("/")[-1] != "billsum" or log.status != "success":
            continue
        arm = arm_of(log)
        if arm is None:
            continue
        # Keep the most recent log per arm, so a re-run supersedes rather than
        # silently competing with what it replaced.
        if arm not in logs or log.eval.created > logs[arm].eval.created:
            logs[arm] = log

    # EVERY POINT ON THE CURVE MUST SHARE ONE GENERATION CAP. The cap truncates
    # long outputs, so an arm scored at 640 tokens and one scored at 1280 are not
    # measuring the same thing, and the difference between them would be read as
    # an effect of training data. This bit us in the obvious way: the base model
    # left 24% of its generations unterminated at 640 and the smallest arm 10%,
    # so the whole curve had to be re-scored at 1280. Refuse to compare across
    # caps rather than quietly averaging them.
    caps = {arm: (log.eval.task_args or {}).get("max_tokens", DEFAULT_MAX_TOKENS)
            for arm, log in logs.items()}
    if len(set(caps.values())) > 1:
        detail = ", ".join(f"{arm}={cap}" for arm, cap in sorted(caps.items()))
        raise SystemExit(
            f"arms were scored under different generation caps ({detail}).\n"
            f"Re-score the odd ones out with -T max_tokens=<the common value> "
            f"before reading this curve."
        )

    missing = [a for a in ARM_ORDER if a not in logs]
    if missing:
        print(f"note: no successful log for {', '.join(missing)}\n")
    present = [a for a in ARM_ORDER if a in logs]
    if not present:
        raise SystemExit(f"no billsum logs in {args.log_dir}")

    samples = {arm: per_sample(logs[arm]) for arm in present}

    width = max(len(m) for m in REPORTED) + 2
    print(f"{'rows':>6}  {'arm':<8}  " + "".join(f"{m:>{width}}" for m in REPORTED))
    for arm in present:
        rows = int(FRACTION[arm] * args.train_size)
        cells = []
        for metric in REPORTED:
            values = samples[arm].get(metric)
            cells.append(f"{statistics.fmean(values.values()):>{width}.4f}"
                         if values else f"{'-':>{width}}")
        print(f"{rows:>6}  {arm:<8}  " + "".join(cells))

    # A cap that truncated most generations would make every number above a
    # measurement of the cap. It cost us weeks on XSum; check it before reading
    # the curve.
    for arm in present:
        unterminated = samples[arm].get("unterminated", {})
        if unterminated and statistics.fmean(unterminated.values()) > 0.1:
            share = statistics.fmean(unterminated.values())
            print(f"\nWARNING: {arm} left {share:.0%} of generations unterminated - "
                  f"raise -T max_tokens before reading this curve.")

    print(f"\npaired bootstrap on {args.metric}, consecutive arms "
          f"({BOOTSTRAP_RESAMPLES} resamples):")
    flat_at_top = None
    for lower, upper in zip(present, present[1:]):
        result = paired_bootstrap(samples[lower].get(args.metric, {}),
                                  samples[upper].get(args.metric, {}))
        if result is None:
            print(f"  {lower:>8} -> {upper:<8}  no shared examples")
            continue
        point, lo, hi, n = result
        flat = lo <= 0 <= hi
        print(f"  {lower:>8} -> {upper:<8}  {point:+.4f}  "
              f"CI [{lo:+.4f}, {hi:+.4f}]  n={n}  {'flat' if flat else 'rising'}")
        if (lower, upper) == ("half", "full"):
            flat_at_top = flat

    if args.json:
        # The .eval logs are the record; this is the summary small enough to
        # commit next to the other results and to read without Inspect.
        payload = {
            "task": "billsum",
            "metric": args.metric,
            "max_tokens": next(iter(set(caps.values()))),
            "test_examples": len(next(iter(samples.values())).get(args.metric, {})),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "note": "one run per arm; seed spread unmeasured. On the crime task it "
                    "was 0.004-0.005, so read any step below that as flat.",
            "arms": [
                {"arm": arm, "rows": int(FRACTION[arm] * args.train_size),
                 **{m: round(statistics.fmean(samples[arm][m].values()), 4)
                    for m in REPORTED if samples[arm].get(m)}}
                for arm in present
            ],
            "steps": [
                {"from": lower, "to": upper, "delta": round(r[0], 4),
                 "ci_low": round(r[1], 4), "ci_high": round(r[2], 4), "n": r[3],
                 "flat": r[1] <= 0 <= r[2]}
                for lower, upper in zip(present, present[1:])
                if (r := paired_bootstrap(samples[lower].get(args.metric, {}),
                                          samples[upper].get(args.metric, {}))) is not None
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")

    # THE QUESTION THE CURVE IS ACTUALLY FOR. A merging experiment reports the
    # recovery fraction R = (model - best shard) / (full - best shard), and that
    # denominator is just the gap between the full-data arm and one shard. An
    # n-way split of the training set puts each shard at 1/n of the data, which
    # is a point this curve already measures - so the curve says, before any
    # expert is trained, whether an n-way experiment has a denominator big
    # enough to divide by.
    SHARDS = {2: "half", 4: "quarter", 8: "eighth"}
    if "full" in samples:
        print("\nrecovery-fraction denominator (full - shard), by shard count:")
        for n, arm in SHARDS.items():
            if arm not in samples:
                continue
            result = paired_bootstrap(samples[arm].get(args.metric, {}),
                                      samples["full"].get(args.metric, {}))
            if result is None:
                continue
            point, lo, hi, _ = result
            # Usable when the interval clears zero AND the gap clears the
            # seed-to-seed spread measured on the crime task, below which a
            # difference cannot be attributed to the method rather than the draw.
            verdict = ("unusable - denominator is noise" if lo <= 0
                       else "marginal" if point < 2 * SEED_SPREAD
                       else "usable")
            print(f"  {n}-way (each expert sees {arm}'s data): {point:+.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    if flat_at_top is True:
        print(
            "\nThe top step is flat: doubling the training data from half to full did not\n"
            "move the score. That is NOT the XSum failure - on XSum nothing separated at all,\n"
            "whereas here base to full is a large, clearly resolved gap. What has flattened\n"
            "is the data axis past the half arm.\n\n"
            "The consequence is specific: a 2-way split puts each expert on the half arm, so\n"
            "the recovery fraction's denominator is the flat step above and R is undefined.\n"
            "Read the shard table - split into more shards, so each expert sits further down\n"
            "the curve where it is still steep, or raise --train-size so a 2-way shard does."
        )
    elif flat_at_top is False:
        print("\nThe top step is rising: the score is still improving at the full "
              "training set.\nBillSum has headroom at this scale, so differences "
              "between merge methods have somewhere\nto show up. Proceed with the "
              "expert/merge arms.")


if __name__ == "__main__":
    main()
