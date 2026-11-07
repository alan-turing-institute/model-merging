# Writes a set of merging results as one markdown report, each arm summarised
# over the seed repeats by its median and interquartile range. The gemma3_4b set
# is the data-parallel merges (crime and XSum, Gemma 3 4B) and the task merges
# (Gemma 3 4B, Mistral 7B, Llama 3.1 8B); the gemma3_1b set, kept in its own
# folder, the data-parallel merges of Gemma 3 1B (crime, wine, XSum, DE->EN).
#
# Usage, from evaluate/:
#   uv run python make_merging_report.py ../results/merging-median-iqr.md
#   uv run python make_merging_report.py --set gemma3_1b ../results/gemma3_1b/merging-median-iqr-1b.md
#   uv run python make_merging_report.py --stat mean mean.md [--tsv cells.tsv]
#
# Every number comes from the per-model JSONs that export_results.py (4B) and
# export_data_parallel.py (1B) write, in the set's results folder (--results
# overrides it), so this runs on any checkout. --stat mean writes the mean ±
# standard deviation instead, which reproduces every row of the set's mean ± SD
# reports. --tsv also writes every distinct cell with the value of each seed.
import argparse
import json
import statistics as st
import subprocess
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
ROOT = RESULTS  # the results folder of the set being reported (main sets it)
METHODS = ["linear", "ties", "dare_ties", "task_arithmetic", "model_stock", "slerp", "arcee_fusion"]
SPLITS = [2, 4, 8]
SEEDS = [1, 2, 3, 4, 5]
# A seed is outlying when it lies more than FENCE IQRs beyond the nearer
# quartile, the IQR being the larger of the cell's own and the median IQR of
# that metric over every arm of the study: with five seeds a cell's own IQR is
# often a unit or two of the last digit, which would make ordinary noise
# outlying.
FENCE = 3
# (JSON metric, column name, decimal places); the first is the task's main metric.
DATA_PARALLEL = {
    "crime": {
        "folder": "crime_data_parallel_gemma", "title": "Crime classification (Reddit posts)", "short": "Crime",
        "name": "crime", "expert": "a LoRA adapter", "main": "accuracy",
        "metrics": [("accuracy/mean", "Accuracy", 3), ("accuracy/f1", "F1", 3)],
        "test": "the 1,077 test posts; accuracy and F1 for the crime class",
    },
    "xsum": {
        "folder": "xsum_data_parallel_gemma", "title": "XSum summarisation (gold targets, long prompt)", "short": "XSum",
        "name": "XSum", "expert": "a student, by self-distillation", "main": "ROUGE-1",
        "metrics": [("rouge1/mean", "ROUGE-1", 3), ("rouge2/mean", "ROUGE-2", 3), ("rougeL/mean", "ROUGE-L", 3),
                    ("pred_words/mean", "Words", 1)],
        "test": "1,000 test articles, long prompt; ROUGE F-measures, and mean summary length in words",
    },
    "wine": {
        "folder": "wine_data_parallel_gemma", "title": "Wine review points (80-100)", "short": "Wine",
        "name": "wine", "expert": "a LoRA adapter", "main": "MAE",
        "metrics": [("abs_error/mean", "MAE", 3), ("abs_error/rmse", "RMSE", 3), ("abs_error/pearson", "Pearson", 3),
                    ("unanswered/mean", "Unanswered", 3)],
        "test": "the 1,000 test reviews; mean absolute and root-mean-square error in points and Pearson correlation "
                "over the answered reviews, and the share with no number in the answer",
    },
    "mt_domain_de_en": {
        "folder": "mt_domain_de_en_data_parallel_gemma", "title": "German-English translation (medical and legal)",
        "short": "DE-EN", "name": "DE-EN translation", "expert": "a LoRA adapter", "main": "chrF++",
        "metrics": [("chrf/mean", "chrF++", 2), ("chrf_medical/mean", "Medical", 2), ("chrf_law/mean", "Legal", 2),
                    ("length_ratio/mean", "Length ratio", 2)],
        "test": "the 2,000 EMEA and JRC-Acquis test pairs (1,000 per domain); mean sentence chrF++, overall and per "
                "domain, and output length over reference length in words",
    },
}
# The sets of results a report covers. The 1B runs are kept apart from the 4B
# ones, in their own results folder.
SETS = {
    "gemma3_4b": {
        "results": RESULTS, "path": "results", "model": "Gemma 3 4B", "title": "Merging results by {stat}", "data_parallel": ["crime", "xsum"],
        "task_merging": True,
        "reports": "`results/crime-data-parallel-gemma.md` and `results/xsum-data-parallel-gemma.md`",
    },
    "gemma3_1b": {
        "results": RESULTS / "gemma3_1b", "path": "results/gemma3_1b", "model": "Gemma 3 1B", "title": "Data-parallel merging by {stat}: Gemma 3 1B",
        "data_parallel": ["crime", "wine", "xsum", "mt_domain_de_en"], "task_merging": False,
        "reports": "`results/gemma3_1b/{crime,wine,xsum,mt-domain-de-en}-data-parallel-gemma-1b.md`",
        "note": "At 1B, Axolotl resizes the embedding to the tokenizer (Gemma 3 1B's tokenizer has one token more than "
                "its embedding), so PEFT saves embed_tokens and lm_head in each LoRA adapter; "
                "`strip_resized_embeddings.py`, run after training, checks that they are the base's plus that row and "
                "drops them, and the LoRA merges name no tokenizer source.",
    },
}
SET = SETS["gemma3_4b"]  # the set being reported (main sets it)
# All three are the instruction-tuned models (their Hub ids are in the headings).
TASK_MODELS = {"gemma": "Gemma 3 4B", "mistral": "Mistral 7B", "llama": "Llama 3.1 8B"}
TASKS = {
    "crime": {"title": "Crime", "metrics": [("accuracy/mean", "Accuracy", 3), ("accuracy/f1", "F1", 3)],
              "test": "1,077 test posts"},
    "wine": {"title": "Wine", "metrics": [("abs_error/mean", "MAE", 3), ("abs_error/rmse", "RMSE", 3),
                                          ("abs_error/pearson", "Pearson", 3)],
             "test": "1,000 test reviews (error in points: lower is better)"},
    "clinc150": {"title": "CLINC150", "metrics": [("accuracy/mean", "Accuracy", 3)],
                 "test": "4,500 test queries, labels prompt"},
    "xsum": {"title": "XSum", "metrics": [("rouge1/mean", "ROUGE-1", 3), ("rouge2/mean", "ROUGE-2", 3),
                                          ("rougeL/mean", "ROUGE-L", 3), ("pred_words/mean", "Words", 1)],
             "test": "1,000 test articles, long prompt"},
}
# The comparisons of merge methods, by their main metric, and the metrics for which lower is better.
VERDICT = {"Accuracy", "ROUGE-1", "MAE", "chrF++", "R all four"}
LOWER = {"MAE", "RMSE"}

cells = {}  # (where, row, column) -> every distinct summarised cell
FLAGS = {}  # (where, row, column) -> indices of its outlying seeds, from a first pass


@cache
def metrics(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["metrics"]


def value(path, key):
    m = metrics(path)
    return None if m is None else m[key]


def num(x, digits):
    s = f"{x:.{digits}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def describe(per_seed):
    vals = [v for v in per_seed if v is not None]
    if not vals:
        return None
    d = {"n": len(vals), "mean": st.mean(vals), "median": st.median(vals)}
    if len(vals) > 1:
        d["sd"] = st.stdev(vals)
        d["q1"], _, d["q3"] = st.quantiles(vals, n=4, method="inclusive")
    return d


def flag():
    """The outlying seeds of every five-seed cell (see FENCE)."""
    def study(c):
        return c["where"].split(",")[0], c["column"]
    full = [c for c in cells.values() if not c["single"] and c["d"] and c["d"]["n"] == len(SEEDS)]
    iqrs = {}
    for c in full:
        iqrs.setdefault(study(c), []).append(c["d"]["q3"] - c["d"]["q1"])
    flags = {}
    for c in full:
        q1, q3 = c["d"]["q1"], c["d"]["q3"]
        reach = FENCE * max(q3 - q1, st.median(iqrs[study(c)]))
        out = [i for i, v in enumerate(c["per_seed"]) if v < q1 - reach or v > q3 + reach]
        if out:
            flags[c["where"], c["row"], c["column"]] = out
    return flags


def fmt(d, digits, stat, single=False, mark=False):
    if d is None:
        return "–"
    short = f", {d['n']} seed{'s' if d['n'] > 1 else ''}" if not single and d["n"] < len(SEEDS) else ""
    if stat == "mean":
        return num(d["mean"], digits) + (f" ± {num(d['sd'], digits)}" if d["n"] > 1 else "") + short
    spread = f"{num(d['q3'] - d['q1'], digits)}{short}" if d["n"] > 1 else short.removeprefix(", ")
    return num(d["median"], digits) + (f" ({spread})" if spread else "") + (" †" if mark else "")


def cell(stat, where, row, column, per_seed, digits, single=False, per_seed_mean=None):
    """Formats one table cell and keeps it, once, for the summary sections. per_seed_mean is the per-seed
    series the mean ± SD reports use, where that differs (the Experts rows)."""
    d = describe(per_seed)
    out = FLAGS.get((where, row, column), [])
    cells.setdefault((where, row, column), {
        "where": where, "row": row, "column": column, "per_seed": per_seed, "digits": digits, "d": d,
        "d_mean": describe(per_seed_mean) if per_seed_mean else d, "outlying": out, "single": single})
    return fmt(d, digits, stat, single, mark=bool(out) and stat == "median")


def head(stat, labels):
    """A table's header row; its corner names the cell format (empty for the mean, as in the mean ± SD reports)."""
    return f"| {'median (IQR) ' if stat == 'median' else ''}| " + " | ".join(labels) + " |"


def data_parallel(task, stat):
    spec = DATA_PARALLEL[task]
    root = ROOT / spec["folder"]
    out = [f"## Data-parallel merging: {spec['title']}, {SET['model']}", "",
           f"The training data is split into N equal parts; an expert, {spec['expert']}, is trained on each part "
           f"and the N experts are merged. Test set: {spec['test']}; the main metric is {spec['main']}"
           f"{' (lower is better)' if spec['metrics'][0][1] in LOWER else ''}. *Experts* is "
           f"the {stat} over the N experts of each seed, then the {stat} over seeds.", ""]
    sep = "|---" * (len(spec["metrics"]) + 2) + "|"
    for n in SPLITS:
        where = f"Data-parallel {spec['short']}, N = {n}"
        out += [f"### {spec['short']}, N = {n}", "", head(stat, [label for _, label, _ in spec["metrics"]] + ["seeds"]),
                sep]

        def row(label, per_seed_of, at=where, per_seed_mean_of=None):
            text, seeds = [], None
            for key, col, digits in spec["metrics"]:
                v = [per_seed_of(s, key) for s in SEEDS]
                vm = [per_seed_mean_of(s, key) for s in SEEDS] if per_seed_mean_of else None
                text.append(cell(stat, at, label, col, v, digits, per_seed_mean=vm))
                seeds = seeds if seeds is not None else sum(x is not None for x in v)
            out.append(f"| {label} | " + " | ".join(text) + f" | {seeds} |")

        out.append("| Base model (untrained) | " + " | ".join(
            cell(stat, f"Data-parallel {spec['short']}", "Base model (untrained)", label,
                 [value(root / "base.json", key)], digits, single=True)
            for key, label, digits in spec["metrics"]) + " | – |")
        # the same five models under every N
        row("**Full data** (one model)", lambda s, key: value(root / f"seed_{s}/full.json", key),
            at=f"Data-parallel {spec['short']}")

        def experts(how):
            def per_seed(s, key):
                vals = [value(root / f"seed_{s}/split_{n}/{i}_of_{n}.json", key) for i in range(1, n + 1)]
                return None if None in vals else how(vals)
            return per_seed
        row(f"Experts (each on 1/{n} of the data)", experts(st.median if stat == "median" else st.mean),
            per_seed_mean_of=experts(st.mean))
        for method in METHODS:
            row(f"merge: {method}", lambda s, key: value(root / f"seed_{s}/split_{n}/merge_{method}.json", key))
        out.append("")
    return out


def task_rows(model):
    """(row label, path of seed S and task T) for every arm of one model's task merges."""
    root = ROOT / "task_merging" / model
    rows = [("Expert (trained on that task)", lambda s, t: root / f"seed_{s}/expert_{t}.json")]
    rows += [(f"merge: {m}", lambda s, t, m=m: root / f"seed_{s}/{m}/{t}.json") for m in METHODS]
    if any((root / f"seed_{s}/moe").exists() for s in SEEDS):
        rows.append(("MoE (mergekit-moe)", lambda s, t: root / f"seed_{s}/moe/{t}.json"))
    return root, rows


def recovery(model):
    """R per row, task and seed: (merged - base) / (expert - base), the expert of the same seed."""
    root, rows = task_rows(model)
    R = {}
    for label, path in rows[1:]:
        for t, spec in TASKS.items():
            key = spec["metrics"][0][0]
            base = value(root / f"base_{t}.json", key)
            R[label, t] = []
            for s in SEEDS:
                merged, expert = value(path(s, t), key), value(rows[0][1](s, t), key)
                R[label, t].append(None if None in (merged, expert, base) else (merged - base) / (expert - base))
        R[label, "all"] = [None if any(R[label, t][i] is None for t in TASKS) else
                           st.mean(R[label, t][i] for t in TASKS) for i in range(len(SEEDS))]
    return R


def model_id(root):
    p = root / "base_crime.json"
    if not p.exists():
        return root.name
    with open(p) as f:
        return json.load(f)["model"].removeprefix("hf/")


def task_merging(model, stat):
    root, rows = task_rows(model)
    where = f"Task merging, {TASK_MODELS[model]}"
    main = [(t, *spec["metrics"][0]) for t, spec in TASKS.items()]
    out = [f"### {TASK_MODELS[model]} (`{model_id(root)}`)", "",
           head(stat, [f"{TASKS[t]['title']} {label}" for t, _, label, _ in main]),
           "|---" * (len(main) + 1) + "|"]
    out.append("| Base model (untrained) | " + " | ".join(
        cell(stat, where, "Base model (untrained)", f"{TASKS[t]['title']} {label}",
             [value(root / f"base_{t}.json", key)], digits, single=True) for t, key, label, digits in main) + " |")
    for label, path in rows:
        out.append(f"| {label} | " + " | ".join(
            cell(stat, where, label, f"{TASKS[t]['title']} {col}", [value(path(s, t), key) for s in SEEDS], digits)
            for t, key, col, digits in main) + " |")
    R = recovery(model)
    out += ["", f"Recovery R ({stat} over seeds):", "",
            head(stat, [TASKS[t]["title"] for t in TASKS] + ["All four"]),
            "|---" * (len(TASKS) + 2) + "|"]
    for label, _ in rows[1:]:
        out.append(f"| {label} | " + " | ".join(
            cell(stat, where, label, f"R {TASKS[t]['title'] if t != 'all' else 'all four'}", R[label, t], 2)
            for t in [*TASKS, "all"]) + " |")
    return out + [""]


def task_appendix(model, task, stat):
    root, rows = task_rows(model)
    spec = TASKS[task]
    where = f"Task merging, {TASK_MODELS[model]}"
    out = [f"### {TASK_MODELS[model]}: {spec['title']}", "",
           head(stat, [label for _, label, _ in spec["metrics"]]),
           "|---" * (len(spec["metrics"]) + 1) + "|"]
    out.append("| Base model (untrained) | " + " | ".join(
        cell(stat, where, "Base model (untrained)", f"{spec['title']} {label}",
             [value(root / f"base_{task}.json", key)], digits, single=True)
        for key, label, digits in spec["metrics"]) + " |")
    for label, path in rows:
        out.append(f"| {label} | " + " | ".join(
            cell(stat, where, label, f"{spec['title']} {col}", [value(path(s, task), key) for s in SEEDS], digits)
            for key, col, digits in spec["metrics"]) + " |")
    return out + [""]


def verdict(stat):
    """For each comparison, the best merges by the median, the methods next to them and the paired wins over them,
    the best by the mean, and the worst. Ties are at the precision shown, for both statistics."""
    groups = {}
    for c in cells.values():
        if c["row"].startswith(("merge", "MoE")) and c["column"] in VERDICT and c["d"] and c["d"]["n"] > 1:
            groups.setdefault(c["where"], []).append(c)

    def name(c):
        return c["row"].removeprefix("merge: ")
    def names(cs):
        return " = ".join(name(c) for c in cs)
    def wins(a, b, sign):
        pairs = [(x, y) for x, y in zip(a["per_seed"], b["per_seed"]) if x is not None and y is not None]
        return f"{sum(sign * (x - y) > 0 for x, y in pairs)} of {len(pairs)}"
    def tiers(cs, k, sign, value):
        """The methods grouped by their value at the precision shown, best first (by the unrounded value)."""
        out = {}
        for c in sorted(cs, key=lambda c: sign * value(c), reverse=True):
            out.setdefault(num(value(c), k), []).append(c)
        return list(out.items())
    rows, same, changed, worst = [], [], [], []
    for where, cs in groups.items():
        if len(cs) < 2:
            continue
        k = cs[0]["digits"]
        sign = -1 if cs[0]["column"] in LOWER else 1  # so that larger sign * value is better
        by_median = tiers(cs, k, sign, lambda c: c["d"]["median"])
        by_mean = tiers(cs, k, sign, lambda c: c["d_mean"]["mean"])
        (top, best), nxt = by_median[0], by_median[1] if len(by_median) > 1 else ("–", [])
        agree = bool({id(c) for c in best} & {id(c) for c in by_mean[0][1]})
        same.append(agree)
        if not agree:
            changed.append(where)
        lows = {id(c): c for c in by_median[-1][1]}, {id(c): c for c in by_mean[-1][1]}
        worst.append(lows)
        low = names(lows[0].values()) + ("" if lows[0].keys() & lows[1].keys() else f" ({names(lows[1].values())} by mean)")
        rows.append(f"| {where} | {names(best)} {top} | {names(nxt[1]) + ' ' + nxt[0] if nxt[1] else '–'} | "
                    + (", ".join(wins(best[0], c, sign) for c in nxt[1]) or "–") + " | "
                    + f"{names(by_mean[0][1])} {by_mean[0][0]} | {'yes' if agree else 'no'} | {low} |")
    if not rows:
        return []
    answer = (f"In {sum(same)} of the {len(same)} comparisons the best merge by the median is also the best by the "
              "mean, ties at the precision shown counting as best"
              + (f"; it changes in {'; '.join(changed)}." if changed else "."))
    # One method worst (or tied worst) in every comparison under both statistics: say so and drop the column.
    common = set.intersection(*({name(c) for c in lo[0].values()} & {name(c) for c in lo[1].values()} for lo in worst))
    if common:
        worst_text = f"{' and '.join(sorted(common))} is the worst merge, or tied worst, in every comparison under both statistics."
        rows = [r.rsplit(" | ", 1)[0] + " |" for r in rows]
    else:
        worst_text = "The worst merge differs between comparisons (last column)."
    mains = [f"{DATA_PARALLEL[t]['main']} ({DATA_PARALLEL[t]['name']}"
             + (", lower is better)" if DATA_PARALLEL[t]["metrics"][0][1] in LOWER else ")") for t in SET["data_parallel"]]
    if SET["task_merging"]:
        mains.append("R over all four tasks (task merging)")
    return ["## Does the summary statistic change the verdict?", "", f"{answer} {worst_text}", "",
            f"The main metric of each comparison is {', '.join(mains[:-1])} and {mains[-1]}. Methods are tied when "
            "they agree at the precision shown, and tied methods are listed together, the better unrounded first. "
            "*Next* is the methods with the second-best median. *Wins* counts the seeds on which the first best "
            "method beats each of them, out of the seeds both have: every method of a seed merges the same experts, "
            "so the comparison is paired. With five seeds neither the medians nor the win counts are a significance "
            "test; they show how consistent a lead is. *Same best* says whether the best by the median and by the "
            "mean have a method in common.", "",
            "| Comparison | Best by median | Next by median | Wins | Best by mean | Same best |"
            + ("" if common else " Worst by median |"), "|---" * (6 if common else 7) + "|", *rows, ""]


def outliers():
    """Five-seed cells with an outlying seed, with the mean ± SD a single seed produces there."""
    order = {w: i for i, w in enumerate(dict.fromkeys(c["where"] for c in cells.values()))}
    task_of = {t["title"]: i for i, t in enumerate(TASKS.values())}
    rows_of = {"Expert (trained on that task)": -1, **{f"merge: {m}": i for i, m in enumerate(METHODS)},
               "MoE (mergekit-moe)": len(METHODS)}

    def task_rank(column):  # "Wine MAE", "R Wine", "R all four", or a data-parallel metric
        words = column.split()
        return task_of.get(words[1] if words[0] == "R" else words[0], len(task_of))
    found = sorted((c for c in cells.values() if c["outlying"]),
                   key=lambda c: (order[c["where"]], task_rank(c["column"]), rows_of.get(c["row"], -2), c["column"]))
    total = sum(1 for c in cells.values() if not c["single"] and c["d"] and c["d"]["n"] == len(SEEDS))
    out = ["## Outlying seeds", "",
           f"Cells in which a seed lies more than {FENCE} IQRs beyond the nearer quartile, taking as the IQR the larger "
           "of the cell's own and the median IQR of that metric over every arm of the study (with five seeds a "
           "cell's own IQR is often a unit or two of the last digit, which would make ordinary noise outlying): "
           f"{len(found)} of the {total} distinct five-seed cells"
           + (", the appendix included" if SET["task_merging"] else "") + ". They are marked † in the tables. The median and IQR barely move for such a seed, which is the point of them, but they also hide "
           "it, and a method that fails on one seed in five is less reliable than its median suggests. The outlying "
           "values are in bold; *Mean ± SD* is computed as in the mean ± SD reports.", "",
           "| Where | Row | Metric | Seeds 1-5 | Mean ± SD | Median (IQR) |", "|---|---|---|---|---|---|"]
    for c in found:
        k = c["digits"]
        vals = ", ".join(f"**{num(v, k)}**" if i in c["outlying"] else num(v, k) for i, v in enumerate(c["per_seed"]))
        out.append(f"| {c['where']} | {c['row']} | {c['column']} | {vals} | {fmt(c['d_mean'], k, 'mean')} | "
                   f"{fmt(c['d'], k, 'median')} |")
    # The same seeds by arm, with the arm's score on its main measure on that seed and on the others.
    arms = {}
    for c in found:
        for i in c["outlying"]:
            arms.setdefault((c["where"], c["row"], i), []).append(c["column"])
    out += ["", "By arm and seed, with the arm's score on that seed and on its other four, on the main metric of its "
            "comparison (as in the verdict above), so that a seed can be judged as a whole: a seed outlying on one "
            "metric may still be weak, or fine, overall.", "",
            "| Where | Row | Seed | Outlying on | Score on that seed | Other seeds |", "|---|---|---|---|---|---|"]
    for (where, row, i), columns in arms.items():
        study = where.split(",")[0]
        main = "R all four" if study == "Task merging" else next(
            spec["metrics"][0][1] for spec in DATA_PARALLEL.values() if study == f"Data-parallel {spec['short']}")
        c = cells.get((where, row, main))
        if c is None or c["per_seed"][i] is None:
            score = others = "–"
        else:
            k = c["digits"]
            rest = [v for j, v in enumerate(c["per_seed"]) if j != i and v is not None]
            score = f"{main} {num(c['per_seed'][i], k)}"
            others = f"{num(min(rest), k)} to {num(max(rest), k)}" if rest else "–"
        out.append(f"| {where} | {row} | {SEEDS[i]} | {', '.join(columns)} | {score} | {others} |")
    return out + [""]


def results_commit():
    folders = [DATA_PARALLEL[t]["folder"] for t in SET["data_parallel"]] + ["task_merging"] * SET["task_merging"]
    git = ["git", "-C", str(ROOT)]
    try:
        commit = subprocess.run([*git, "log", "-1", "--format=%h", "--", *folders], capture_output=True, text=True)
        dirty = subprocess.run([*git, "status", "--porcelain", "--", *folders], capture_output=True, text=True).stdout
    except OSError:  # no git
        return ""
    if commit.returncode or not commit.stdout.strip():
        return " (not yet committed)" if dirty.strip() else ""
    return f" (as of commit {commit.stdout.strip()}{', with uncommitted changes' if dirty.strip() else ''})"


def build(stat):
    title = "median and interquartile range" if stat == "median" else "mean and standard deviation"
    tasks, tm = SET["data_parallel"], SET["task_merging"]
    names = [DATA_PARALLEL[t]["name"] for t in tasks]
    out = [f"# {SET['title'].format(stat=title)}", "",
           f"Generated {datetime.now(timezone.utc):%Y-%m-%d} by `evaluate/make_merging_report.py` from the per-model "
           f"JSONs in `{SET['path']}/`{results_commit()}.", ""]
    if stat == "median":
        out += [f"Contents: data-parallel merging of {', '.join(names[:-1])} and {names[-1]} ({SET['model']}); "
                + (f"task merging ({', '.join(TASK_MODELS.values())}); " if tm else "")
                + "whether the summary statistic changes the verdict; outlying seeds"
                + ("; an appendix with every task-merging metric." if tm else "."), "",
                "Each cell is the median over seeds 1-5 with the interquartile range in brackets, *median (IQR)*, as "
                "each table's corner says; the base model is evaluated once. Quartiles interpolate linearly between "
                "the sorted values (Python's `statistics.quantiles` with `method=\"inclusive\"`, numpy's default), so "
                "with five seeds Q1 and Q3 are the second and fourth smallest values and the IQR, Q3 - Q1, is the "
                "full width of the middle three seeds, not a ± half-width. For five seeds that scatter normally it "
                "averages 0.99 standard deviations, about 1.05 times the sample SD the mean ± SD reports give. That is "
                "only 0.73 of the IQR of the distribution the seeds come from (1.35 standard deviations), so these "
                "IQRs understate the spread, by the same factor in every five-seed cell, which leaves comparisons "
                "between arms unaffected. Q1 and Q3 are single seeds, so the IQR is itself noisy, and (0.000) means "
                "only that the second and fourth seeds agree at the precision shown.", "",
                "One seed that fails can move the median and IQR only as far as a neighbouring seed's value, whereas "
                "it can move the mean and standard deviation without limit. The price is that the median and IQR "
                "ignore the best and worst seed, so a method that fails on one seed in five can look reliable: cells "
                "with such a seed are marked † and listed under *Outlying seeds*.", "",
                f"The data-parallel tables are also in {SET['reports']} as mean ± SD. Those are built from the `.eval` "
                "logs, and these JSONs reproduce every row of them (`--stat mean`), so any difference comes from the "
                "statistic alone."
                + (" Task merging has no mean ± SD report; `make_merging_report.py --stat mean` writes this one that "
                   "way." if tm else ""), ""]
    else:
        out += ["Each cell is the mean over seeds 1-5 ± the standard deviation; the base model is evaluated once.", ""]
    out += ["Every arm is repeated with seeds 1-5. The seed sets the training data order (for the LoRA experts also "
            "the adapter initialisation and dropout) and mergekit's random seed, which dare_ties uses to choose the "
            "weights it drops and arcee_fusion to sample the tensors its thresholds are estimated from.", "",
            "Merge settings (`merge/make_merge_configs.py`" + (", `merge/make_task_merge_configs.py`" if tm else "")
            + "): linear, ties, dare_ties and task_arithmetic give each of the N models merged weight 1/N (N = 2, 4 "
            "or 8 in data-parallel merging" + (", 4 in task merging" if tm else "") + "); ties density 0.5; dare_ties "
            "density 0.3; model_stock takes no weights. slerp (t = 0.5) and arcee_fusion take two models, so they are "
            "applied as a binary tree of pairwise merges"
            + (" (for task merging, (crime + wine) + (CLINC150 + XSum))" if tm else "")
            + ", with the first model of each pair as mergekit's base model; arcee_fusion keeps that base and adds "
            "the other model's selected changes, so it is not symmetric. With weights summing to 1 on a shared base, "
            "linear and task_arithmetic are the same model in exact arithmetic, so any difference between their rows "
            "is bf16 rounding and its effect on greedy decoding." + (f" {SET['note']}" if "note" in SET else ""), ""]
    for task in tasks:
        out += data_parallel(task, stat)
    appendix = []
    if tm:
        out += ["## Task merging", "",
                "Four experts, one per task, each trained on all of its task's data (MLP-only: LoRA for crime, wine "
                "and CLINC150; self-distillation for XSum), are merged into one model, which is evaluated on all four "
                "tasks. Each method merges the same four experts. The MoE (Mistral and Llama only: mergekit has no MoE "
                "path for Gemma) routes each token, at every layer, to one expert's MLP; its attention, embeddings and "
                "norms are the base model's. Test sets: "
                + "; ".join(f"{s['title']}, {s['test']}" for s in TASKS.values()) + ".", "",
                "*Recovery R* is the share of the expert's gain over the base model that a merge keeps, for one task "
                "and seed: (merge - base) / (expert - base), with the expert of the same seed, on the task's main "
                "metric (crime accuracy, wine MAE, CLINC150 accuracy, XSum ROUGE-1: the columns of each model's first "
                "table). 1 is as good as the expert, 0 no better than the untrained "
                f"model, and below 0 worse than it. *All four* is the mean of a seed's four R values, then the {stat} "
                "over seeds; it is a mean because it is a score, and a median of four would ignore the best and the "
                "worst task.", ""]
        for model in TASK_MODELS:
            out += task_merging(model, stat)
        appendix = ["## Appendix: every metric, task merging", ""]
        for model in TASK_MODELS:
            for task in TASKS:
                appendix += task_appendix(model, task, stat)
    out += verdict(stat)
    if stat == "median":
        out += outliers()
    return out + appendix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--stat", choices=["median", "mean"], default="median")
    p.add_argument("--set", choices=SETS, default="gemma3_4b", help="the set of results (default %(default)s)")
    p.add_argument("--results", type=Path, help="the set's results folder, if not where SETS says")
    p.add_argument("--tsv", help="also write every distinct cell, with each seed's value, to this file; an Experts "
                                 "row's seed values are the --stat over the N experts, its mean and sd the originals'")
    args = p.parse_args()
    stat = args.stat
    global SET, ROOT
    SET = SETS[args.set]
    ROOT = (args.results or SET["results"]).resolve()
    needed = [ROOT / DATA_PARALLEL[t]["folder"] / "base.json" for t in SET["data_parallel"]]
    needed += [ROOT / "task_merging"] if SET["task_merging"] else []
    if missing := [str(n) for n in needed if not n.exists()]:
        p.error(f"no {', '.join(missing)}: is --results the {args.set} results folder?")
    if stat == "median":  # a first pass finds the outlying seeds, which the tables then mark
        build(stat)
        FLAGS.update(flag())
        cells.clear()
    out = build(stat)
    Path(args.out).write_text("\n".join(out).rstrip() + "\n")
    if args.tsv:
        lines = ["where\trow\tmetric\t" + "\t".join(f"seed_{s}" for s in SEEDS) + "\tmean\tsd\tmedian\tq1\tq3"]
        for c in cells.values():
            # the base model's single evaluation has no seed
            vals = [None] * len(SEEDS) if c["single"] else c["per_seed"]
            stats = [(c["d_mean"] or {}).get(k) for k in ("mean", "sd")] + [(c["d"] or {}).get(k)
                                                                             for k in ("median", "q1", "q3")]
            lines.append("\t".join([c["where"], c["row"], c["column"]]
                                   + ["" if v is None else repr(v) for v in vals + stats]))
        Path(args.tsv).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
