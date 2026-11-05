# Writes the task-merging results as a markdown report: for each model (Gemma 3
# 4B, Mistral 7B, Llama 3.1 8B), the untrained base model, the MLP-only expert
# for each task, and the merges of the four experts by each dense method (and,
# for Mistral and Llama, the Mixtral-style MoE), evaluated on all four tasks,
# over seeds 1-5.
#
# Usage, from evaluate/:
#   python make_task_merge_tables.py ../results/task_merging ../results/task-merging.md
#
# Every number comes from the JSON exports in results/task_merging/ (made by
# eval_to_json.py; index.tsv there maps each to its .eval log):
#   <model>/base_<task>.json, <model>/seed_<S>/expert_<task>.json (own task),
#   <model>/seed_<S>/expert_<expert>_on_<task>.json (the other tasks),
#   <model>/seed_<S>/<method>/<task>.json
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

MODELS = [("gemma", "Gemma 3 4B (google/gemma-3-4b-it)"),
          ("mistral", "Mistral 7B (mistralai/Mistral-7B-Instruct-v0.3)"),
          ("llama", "Llama 3.1 8B (meta-llama/Llama-3.1-8B-Instruct)")]
# task, column heading, metric key, digits, higher is better
TASKS = [("crime", "Crime acc ↑", "accuracy/mean", 3, True),
         ("wine", "Wine MAE ↓", "abs_error/mean", 2, False),
         ("clinc150", "CLINC150 acc ↑", "accuracy/mean", 3, True),
         ("xsum", "XSum ROUGE-1 ↑", "rouge1/mean", 3, True)]
TASK_NAMES = {"crime": "crime", "wine": "wine", "clinc150": "CLINC150", "xsum": "XSum"}
METHODS = ["linear", "ties", "dare_ties", "task_arithmetic", "model_stock", "slerp", "arcee_fusion", "moe"]
SEEDS = [1, 2, 3, 4, 5]


def load(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d["metrics"]


# A wine cell is flagged † when, averaged over seeds, more than this share of
# answers was out of the 80-100 range or had no number: its MAE is then over a
# different, and partly absurd, set of answers (e.g. a vintage year, "2017").
WINE_FLAG = 0.02


def fmt(vals, digits):
    if not vals:
        return "–"
    sd = f" ± {st.stdev(vals):.{digits}f}" if len(vals) > 1 else ""
    return f"{st.mean(vals):.{digits}f}{sd}"


def wine_flags(d, names):
    """Per seed, the share of wine answers out of range or with no number."""
    out = []
    for s in SEEDS:
        for n in names:
            m = load(d / f"seed_{s}" / n)
            if m:
                out.append(m["out_of_range/mean"] * (1 - m["unanswered/mean"]) + m["unanswered/mean"])
    return out


def section(root, model, title):
    d = root / model
    base = {t: load(d / f"base_{t}.json")[k] for t, _, k, *_ in TASKS}
    bm = load(d / "base_wine.json")
    bw = bm["out_of_range/mean"] * (1 - bm["unanswered/mean"]) + bm["unanswered/mean"]
    expert = {(s, t): (load(d / f"seed_{s}" / f"expert_{t}.json") or {}).get(k)
              for s in SEEDS for t, _, k, *_ in TASKS}
    out = [f"## {title}", "",
           "| | " + " | ".join(h for _, h, *_ in TASKS) + " | Retention | seeds |",
           "|---" * (len(TASKS) + 3) + "|",
           "| Base model (untrained) | " + " | ".join(fmt([base[t]], dg) + (f" † ({bw:.0%})" if t == "wine" and bw > WINE_FLAG else "")
                                                  for t, _, _, dg, _ in TASKS) + " | 0 | – |"]

    def row(label, per_seed, wine_flags=None):  # per_seed: [{task: value}] or None per seed
        cells, rets = [], []
        for t, _, _, dg, _ in TASKS:
            cells.append(fmt([v[t] for v in per_seed if v and v.get(t) is not None], dg))
            if t == "wine" and wine_flags and st.mean(wine_flags) > WINE_FLAG:
                cells[-1] += f" † ({st.mean(wine_flags):.0%})"
        # retention, per seed: mean over the tasks of (x - base) / (expert - base)
        for s, v in zip(SEEDS, per_seed):
            if v and all(v.get(t) is not None for t, *_ in TASKS):
                rets.append(st.mean((v[t] - base[t]) / (expert[(s, t)] - base[t]) for t, *_ in TASKS))
        n = sum(1 for v in per_seed if v)
        out.append(f"| {label} | " + " | ".join(cells) + f" | {fmt(rets, 2)} | {n} |")

    for e, *_ in TASKS:  # each expert on its own task, then on the others
        per_seed = []
        for s in SEEDS:
            vals = {t: expert[(s, t)] if t == e else
                    (load(d / f"seed_{s}" / f"expert_{e}_on_{t}.json") or {}).get(k) for t, _, k, *_ in TASKS}
            per_seed.append(vals if any(v is not None for v in vals.values()) else None)
        row(f"Expert: {TASK_NAMES[e]}", per_seed, wine_flags(d, [f"expert_{e}_on_wine.json" if e != "wine"
                                                                 else "expert_wine.json"]))
    for m in METHODS:
        per_seed = []
        for s in SEEDS:
            vals = {t: (load(d / f"seed_{s}" / m / f"{t}.json") or {}).get(k) for t, _, k, *_ in TASKS}
            per_seed.append(vals if any(v is not None for v in vals.values()) else None)
        if any(per_seed):
            row("MoE (Mixtral, 4 experts, top-1)" if m == "moe" else f"merge: {m}", per_seed,
                wine_flags(d, [f"{m}/wine.json"]))
    out.append("")
    return out


def report(root):
    root = Path(root)
    out = ["# Task merging: crime, wine, CLINC150 and XSum experts merged into one model", "",
           f"Generated {datetime.now(timezone.utc):%Y-%m-%d} by `evaluate/make_task_merge_tables.py` from the JSON "
           "results in `results/task_merging/` (`index.tsv` there maps each file to its `.eval` log in "
           "`evaluate/logs/`).", "",
           "For each base model, one expert is trained per task, with LoRA on the MLP layers only (crime, wine, "
           "CLINC150) or full self-distillation of the MLP layers on gold targets (XSum); the four experts are then "
           "merged into one model. Every arm is repeated with seeds 1-5 (expert training, and dare_ties' random "
           "dropping), so each cell is the mean ± standard deviation over the seeds; the base model is evaluated "
           "once.", "",
           "Test sets and metrics: crime, 1,077 Reddit posts, accuracy; wine, 1,000 Wine Enthusiast reviews, mean "
           "absolute error of the predicted points (lower is better); CLINC150, 4,500 queries with the labels "
           "prompt, accuracy; XSum, 1,000 BBC articles with the long prompt, ROUGE-1 F-measure against the gold "
           "summary. Greedy decoding throughout.", "",
           "Each expert is evaluated on all four tasks: its own (the diagonal) and the three it was not trained on "
           "(`evaluate/submit_expert_cross_evals.sh`, run 5 Nov). An expert's retention is how much of the four "
           "experts' combined gain over the base model it alone gives, so about 0.25 plus whatever it transfers. "
           "**Retention** is, for each seed, the mean over the four tasks of (merged − base) / (expert − base), "
           "using that seed's expert for each task: 1 means the merge matches every expert on its own task, 0 means "
           "no better than the base model, and negative means worse than the base. It is then averaged over seeds.",
           "",
           f"**†** Wine MAE is over the answers that contain a number (`evaluate/wine_task.py`). A wine cell is "
           f"marked † (with the share in brackets) when over {WINE_FLAG:.0%} of answers, averaged over seeds, had "
           "no number or a number outside 80-100. This happens to experts off their own task: they often answer "
           "with the vintage year (\"2017\", an error of ~1,930 points) or with text (\"not yet rated\"), so the "
           "MAE, its spread and that row's retention are driven by a few absurd answers and say little about skill "
           "at rating wine.",
           "",
           "Merge settings (`merge/make_task_merge_configs.py`, the same as for data-parallel merging): weight 1/4 "
           "for every expert; ties density 0.5; dare_ties density 0.3; slerp (t = 0.5) and arcee_fusion take two "
           "models, so they are applied as a binary tree: (crime, wine) and (CLINC150, XSum), then the two results. "
           "The MoE (`merge/make_moe_configs.py`, Mistral and Llama only, since mergekit-moe has no Gemma 3 path) "
           "keeps each expert's MLP as one Mixtral expert on the untrained base, with `gate_mode: hidden` routing "
           "from task prompts and one expert per token; its XSum evaluations ran at batch size 2 rather than 8 "
           "(out of GPU memory), apart from Mistral and Llama seed 3.", ""]
    for model, title in MODELS:
        out += section(root, model, title)
    return "\n".join(out)


if __name__ == "__main__":
    text = report(sys.argv[1])
    Path(sys.argv[2]).write_text(text)
    print(text)
