# Writes the data-parallel merging results for one task as a markdown report:
# for each split N (2, 4, 8), the untrained base model, the model trained on all
# the data, the N experts each trained on 1/N of it, and the merges of those
# experts by each method, over the seed repeats (models/<task>/seed_<S>/, or
# models/<task>_<size>/seed_<S>/ for a base other than gemma-3-4b-it).
#
# Usage, from evaluate/:
#   uv run python make_data_parallel_tables.py crime ../results/crime-data-parallel-gemma.md
#   uv run python make_data_parallel_tables.py xsum ../results/xsum-data-parallel-gemma.md
#   uv run python make_data_parallel_tables.py wine ../results/wine-data-parallel-gemma-1b.md \
#       --base google/gemma-3-1b-it
#
# Every number comes from the .eval logs in logs/: a log is matched to its model
# through the model_provenance path that evaluate.sbatch records in its
# metadata, and where a model was evaluated more than once the latest
# successful log counts. XSum is the gold-target, long-prompt students
# (gemma3_<size>_xsum_gold_*_long), evaluated with the long prompt they were
# trained on; the base model with the same prompt. Translation is German->English
# medical/legal, with the prompt it was trained on.
import argparse
import re
import statistics as st
from datetime import datetime, timezone

from inspect_ai.log import list_eval_logs, read_eval_log

METHODS = ["linear", "ties", "dare_ties", "task_arithmetic", "model_stock", "slerp", "arcee_fusion"]
SPLITS = [2, 4, 8]
SEEDS = [1, 2, 3, 4, 5]
BASE = "google/gemma-3-4b-it"
# Model names carry {size}, the base model's size (4b, 1b). A metric is
# (score, metric, label[, decimals]); three decimals unless given.
TASKS = {
    "crime": {
        "title": "Crime classification (Reddit posts)",
        "metrics": [("accuracy", "mean", "Accuracy"), ("accuracy", "f1", "F1")],
        "full": r"gemma3_{size}_crime", "part": r"gemma3_{size}_crime_(\d)_of_(\d)",
        "merge": r"gemma3_{size}_crime_merge_(\d)_(\w+)",
        "prompt": None, "method": "LoRA",
        "test": "the 1,077 test posts; accuracy and F1 for the crime class",
    },
    "wine": {
        "title": "Wine review points (80-100)",
        "metrics": [("abs_error", "mean", "MAE"), ("abs_error", "rmse", "RMSE"), ("abs_error", "pearson", "Pearson"),
                    ("unanswered", "mean", "Unanswered")],
        "full": r"gemma3_{size}_wine", "part": r"gemma3_{size}_wine_(\d)_of_(\d)",
        "merge": r"gemma3_{size}_wine_merge_(\d)_(\w+)",
        "prompt": None, "method": "LoRA",
        "test": "the 1,000 test reviews; mean absolute and root-mean-square error in points and Pearson correlation "
                "over the answered reviews, and the share with no number in the answer",
    },
    "xsum": {
        "title": "XSum summarisation (gold targets, long prompt)",
        "metrics": [("rouge1", "mean", "ROUGE-1"), ("rouge2", "mean", "ROUGE-2"), ("rougeL", "mean", "ROUGE-L"),
                    ("pred_words", "mean", "Words", 1)],
        "full": r"gemma3_{size}_xsum_gold_long", "part": r"gemma3_{size}_xsum_gold_(\d)_of_(\d)_long",
        "merge": r"gemma3_{size}_xsum_gold_long_merge_(\d)_(\w+)",
        "prompt": "long", "method": "self-distillation",
        "test": "1,000 test articles, long prompt; ROUGE F-measures, and mean summary length in words",
    },
    "mt_domain_de_en": {
        "title": "German-English translation (medical and legal)",
        "metrics": [("chrf", "mean", "chrF++", 2), ("chrf_medical", "mean", "Medical", 2),
                    ("chrf_law", "mean", "Legal", 2), ("length_ratio", "mean", "Length ratio", 2)],
        "full": r"gemma3_{size}_mt_domain_de_en", "part": r"gemma3_{size}_mt_domain_de_en_(\d)_of_(\d)",
        "merge": r"gemma3_{size}_mt_domain_de_en_merge_(\d)_(\w+)",
        "prompt": None, "method": "LoRA",
        "test": "the EMEA and JRC-Acquis test pairs; mean sentence chrF++, overall and per domain, and output "
                "length over reference length in words",
    },
}


def collect(task, base, folder):
    spec = TASKS[task]
    found = {}  # (seed or None, model name) -> (created, {metric: value})
    for info in list_eval_logs("logs"):
        # Inspect writes the task's name into the log's with hyphens.
        if f"_{task.replace('_', '-')}_" not in info.name:
            continue
        h = read_eval_log(info, header_only=True)
        if h.status != "success" or h.eval.task != task:
            continue
        if spec["prompt"] and (h.eval.task_args or {}).get("prompt_variant") != spec["prompt"]:
            continue
        values = {(s.name, k): m.value for s in h.results.scores for k, m in s.metrics.items()}
        md = h.eval.metadata or {}
        m = re.search(rf"models/{folder}/seed_(\d)/(?:split_\d/)?([^/]+)/provenance\.json", md.get("model_provenance") or "")
        if m:
            key = (int(m.group(1)), m.group(2))
        elif (h.eval.model == f"hf/{base}" and md.get("model_kind") in ("base", None)
              and not {"adapter", "model_path"} & set(h.eval.model_args or {})):
            # The base model alone; logs from before evaluate.sbatch recorded
            # model_kind (e.g. XSum's of 12 Oct, on the same 1,000 test
            # articles) have no metadata, and no adapter either.
            key = (None, "base")
        else:
            continue
        if key not in found or h.eval.created > found[key][0]:
            found[key] = (h.eval.created, values)
    return {k: v for k, (_, v) in found.items()}


def fmt(vals, metric):
    if not vals:
        return "–"
    digits = metric[3] if len(metric) > 3 else 3
    mean = st.mean(vals)
    sd = f" ± {st.stdev(vals):.{digits}f}" if len(vals) > 1 else ""
    return f"{mean:.{digits}f}{sd}"


def report(task, base_model):
    spec = TASKS[task]
    size = re.fullmatch(r"google/gemma-3-(\d+b)-it", base_model).group(1)
    folder = task + ("" if base_model == BASE else f"_{size}")
    spec = {**spec, **{k: spec[k].replace("{size}", size) for k in ("full", "part", "merge")}}
    res = collect(task, base_model, folder)
    head = "| | " + " | ".join(m[2] for m in spec["metrics"]) + " | seeds |"
    sep = "|---" * (len(spec["metrics"]) + 2) + "|"
    out = [f"# Data-parallel merging: {spec['title']}, Gemma 3 {size.upper()}", "",
           f"Generated {datetime.now(timezone.utc):%Y-%m-%d} by `evaluate/make_data_parallel_tables.py` from the "
           "`.eval` logs in `evaluate/logs/`.", "",
           f"The training data is split into N equal parts; an expert is trained on each part ({spec['method']}, "
           f"from {base_model}) and the N experts are merged. Every arm is repeated with seeds 1-5 "
           "(training data order and initialisation, and dare_ties' random dropping), so each cell is the mean "
           "± standard deviation over the seeds; the base model is evaluated once. "
           f"Test set: {spec['test']}.", "",
           "Merge settings (`merge/make_merge_configs.py`): weight 1/N for every expert; ties density 0.5; "
           "dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two models, so they are applied as a "
           "binary tree of pairwise merges. *Experts* is the mean over the N experts of each seed, then over seeds.", ""]
    base = res.get((None, "base"))
    for n in SPLITS:
        out += [f"## N = {n}", "", head, sep]
        def row(label, per_seed_values):
            cells = []
            for metric in spec["metrics"]:
                vals = [v[metric[:2]] for v in per_seed_values if v and metric[:2] in v]
                cells.append(fmt(vals, metric))
            k = sum(1 for v in per_seed_values if v)
            out.append(f"| {label} | " + " | ".join(cells) + f" | {k} |")
        out.append("| Base model (untrained) | " + " | ".join(
            fmt([base[m[:2]]], m) if base and m[:2] in base else "–" for m in spec["metrics"]) + " | – |")
        row("**Full data** (one model)", [res.get((s, spec["full"])) for s in SEEDS])
        # experts: per seed, the mean over its N parts (only seeds with all N)
        experts = []
        for s in SEEDS:
            parts = [v for (seed, name), v in res.items() if seed == s
                     and (m := re.fullmatch(spec["part"], name)) and int(m.group(2)) == n]
            experts.append({metric[:2]: st.mean(p[metric[:2]] for p in parts) for metric in spec["metrics"]}
                           if len(parts) == n else None)
        row(f"Experts (each on 1/{n} of the data)", experts)
        for method in METHODS:
            name = re.sub(r"\(\\d\)_\(\\w\+\)$", f"{n}_{method}", spec["merge"])
            row(f"merge: {method}", [res.get((s, name)) for s in SEEDS])
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("path", help="the markdown report to write")
    ap.add_argument("--base", default=BASE, help="the base model the experts were trained from (default %(default)s)")
    args = ap.parse_args()
    if not re.fullmatch(r"google/gemma-3-\d+b-it", args.base):
        ap.error(f"--base must be google/gemma-3-<size>-it, not {args.base}")
    text = report(args.task, args.base)
    open(args.path, "w").write(text)
    print(text)
