"""Tables of the LLM judge and BERTScore on the summarisation merge experiments.

    python summ_report.py ../results/summ_metrics [--judge judge4-qwen3-32b] [--md ../results/summ-judge-merges.md]

Reads each task's summary.json (written by summ_metrics.py) and prints, per
experiment, the untrained model, the reference-trained and full-data models,
the two halves and the better half per seed, and each merge method, every
row a mean over the three seeds where there are seeds (with the SD across them).
"""

import argparse
import json
import statistics
from pathlib import Path

SEEDS = ["", "_s2", "_s3"]
METHODS = ["linear", "ties", "dare_ties", "task_arithmetic"]
COLS = [("bertscore", "bertscore_f1", "BERTScore F1"), ("judge", "coherence", "coherence"),
        ("judge", "consistency", "consistency"), ("judge", "fluency", "fluency"),
        ("judge", "relevance", "relevance"), ("judge", "judge_mean", "judge mean"),
        ("judgelen", "words", "words"), ("judgelen", "relevance_len", "relevance (length)"),
        ("judgelen", "judge_mean_len", "judge mean (length)"),
        ("deepeval", "deepeval", "DeepEval"), ("deepeval", "alignment", "DE alignment"),
        ("deepeval", "coverage", "DE coverage")]


def experiments():
    def kd(task, sft):
        p = f"gemma3-{task}-kdgold75_"
        return {"task": task, "base": "base-gemma-3-4b-it", "sft": sft,
                "full": "gemma3-xsum_kd_gold_ce75-lora" if task == "xsum" else f"{p}full-lora",
                "halves": (lambda s: f"{p}a{s}-lora", lambda s: f"{p}b{s}-lora"),
                "merged": lambda m, s: f"{task}-kdgold75-merged-{m}{s}"}
    return {
        "XSum, gold-target KD": kd("xsum", "gemma3-xsum_screen-full-lora"),
        "CNN/DailyMail, gold-target KD": kd("cnndm", "gemma3-cnndm_curve-n500-lora"),
        "SAMSum, gold-target KD": kd("samsum", "gemma3-samsum-n800-lora"),
        "BillSum, gold-target KD": kd("billsum_split", None),
        "BillSum, reference-trained (SFT)": {
            "task": "billsum_split", "base": "base-gemma-3-4b-it", "sft": None,
            "full": "gemma3-billsum_split-full-lora",
            "halves": (lambda s: f"gemma3-billsum_split-expert_1{s}-lora",
                       lambda s: f"gemma3-billsum_split-expert_2{s}-lora"),
            "merged": lambda m, s: f"billsum_split-merged-{m}{s}"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, nargs="?", default=Path("../results/summ_metrics"))
    ap.add_argument("--judge", default="judge4-qwen3-32b")
    ap.add_argument("--bertscore", default="bertscore-roberta-large")
    ap.add_argument("--deepeval", default="deepeval-qwen3-32b-n5")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    tags = {"judge": args.judge, "bertscore": args.bertscore, "deepeval": args.deepeval,
            "judgelen": f"{args.judge}-len"}
    out = []
    for title, ex in experiments().items():
        summary = json.loads((args.root / ex["task"] / "summary.json").read_text())

        def val(label, col):
            group, key, _ = col
            return ((summary.get(label) or {}).get(tags[group]) or {}).get(key)

        def row(name, labels_by_seed, combine=None):
            cells = []
            for col in COLS:
                vals = [combine(col, s) if combine else val(labels_by_seed(s), col) for s in SEEDS]
                vals = [v for v in vals if v is not None]
                if not vals:
                    cells.append("-")
                elif len(vals) == 1:
                    cells.append(f"{vals[0]:.3f}")
                else:
                    cells.append(f"{statistics.fmean(vals):.3f} ±{statistics.stdev(vals):.3f}")
            return f"| {name} | " + " | ".join(cells) + " |"

        single = lambda label: (lambda s: label if s == "" else None)
        a, b = ex["halves"]
        lines = [f"### {title}", "", "| model | " + " | ".join(c[2] for c in COLS) + " |",
                 "|---|" + "---|" * len(COLS),
                 row("untrained", single(ex["base"]))]
        if ex["sft"]:
            lines.append(row("reference-trained full", single(ex["sft"])))
        lines += [row("full data", single(ex["full"])), row("half A", a), row("half B", b),
                  row("better half", None, lambda col, s: max(
                      (v for v in (val(a(s), col), val(b(s), col)) if v is not None), default=None))]
        lines += [row(f"merged: {m}", lambda s, m=m: ex["merged"](m, s)) for m in METHODS]
        out += lines + [""]
    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.write_text(f"Judge `{args.judge}`, BERTScore `{args.bertscore}`, DeepEval `{args.deepeval}` "
                           "(CNN/DailyMail and BillSum only); '(length)' columns cap relevance outside the 10th-90th "
                           "percentile of the task's reference lengths (summ_judge.py LENGTH PENALTY); "
                           "means over seeds ± SD.\n\n" + text)


if __name__ == "__main__":
    main()
