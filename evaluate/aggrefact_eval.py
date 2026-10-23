"""Balanced accuracy of DeepEval's alignment and the LLM judge's consistency on
LLM-AggreFact's summarisation subsets, beside the benchmark's leaderboard.

    uv run --no-sync python aggrefact_eval.py [--root ../results/summ_metrics_aggrefact] [--md ../results/aggrefact.md]

Protocol of Tang, Laban & Durrett (2024): balanced accuracy (mean of the true
positive and true negative rates) on the test split, with the decision
threshold at the midpoint of the checker's score range and no tuning - judge
consistency (1-5) >= 3, DeepEval alignment (0-1) >= 0.5. Also reported, as in
the paper's appendix: the threshold that maximises balanced accuracy on the dev
split, applied to test. Leaderboard values: llm-aggrefact.github.io.
"""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
LEADERBOARD = {  # (AggreFact-CNN, AggreFact-XSum) balanced accuracy, test split
    "GPT-4o (2024-05-13)": (68.1, 76.8), "Claude-3.5 Sonnet": (67.6, 75.1),
    "Llama-3.3-70B-Instruct": (68.7, 74.7), "Qwen2.5-72B-Instruct": (63.6, 73.0),
    "QwQ-32B-Preview": (57.0, 71.6), "MiniCheck-Flan-T5-L (0.8B)": (69.9, 74.3),
    "Bespoke-MiniCheck-7B": (65.5, 77.8),
}
CHECKERS = {  # name: (record tag, field, midpoint threshold)
    "LLM judge consistency (Qwen3-32B)": ("judge4-qwen3-32b", "consistency", 3.0),
    "DeepEval alignment (Qwen3-32B)": ("deepeval-qwen3-32b-n5", "alignment", 0.5),
    "DeepEval score (Qwen3-32B)": ("deepeval-qwen3-32b-n5", "deepeval", 0.5),
}


def records(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("error") is None:
                out[r["id"]] = r
    return out


def bacc(pairs, threshold):
    pos = [s >= threshold for s, y in pairs if y == 1]
    neg = [s < threshold for s, y in pairs if y == 0]
    return 100 * (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=HERE.parent / "results" / "summ_metrics_aggrefact")
    ap.add_argument("--data", type=Path, default=HERE.parent / "datasets" / "aggrefact")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    out = ["| checker | AggreFact-CNN | AggreFact-XSum | average | CNN, dev-tuned | XSum, dev-tuned | scored (CNN / XSum) |",
           "|---|---|---|---|---|---|---|"]
    report = {}
    for name, (tag, field, mid) in CHECKERS.items():
        cells, tuned, n = [], [], []
        for subset in ("cnn", "xsum"):
            pairs = {}
            for split in ("dev", "test"):
                labels = {r["id"]: r["label"] for r in load_from_disk(str(args.data / f"{subset}_{split}"))}
                recs = records(args.root / f"aggrefact_{subset}" / f"aggrefact_{subset}-{split}.{tag}.jsonl")
                pairs[split] = [(recs[i][field], labels[i]) for i in labels if i in recs and recs[i].get(field) is not None]
            if not pairs["test"]:
                cells.append(None); tuned.append(None); n.append("0"); continue
            cells.append(bacc(pairs["test"], mid))
            if pairs["dev"]:
                cands = sorted({s for s, _ in pairs["dev"]})
                best = max(cands, key=lambda t: bacc(pairs["dev"], t))
                tuned.append(bacc(pairs["test"], best))
            else:
                tuned.append(None)
            n.append(str(len(pairs["test"])))
        f = lambda x: "-" if x is None else f"{x:.1f}"
        avg = None if None in cells else sum(cells) / 2
        out.append(f"| **{name}** | {f(cells[0])} | {f(cells[1])} | {f(avg)} | {f(tuned[0])} | {f(tuned[1])} | {' / '.join(n)} |")
        report[name] = {"cnn": cells[0], "xsum": cells[1], "cnn_dev_tuned": tuned[0], "xsum_dev_tuned": tuned[1]}
    for name, (c, x) in LEADERBOARD.items():
        out.append(f"| {name} (leaderboard) | {c:.1f} | {x:.1f} | {(c + x) / 2:.1f} | | | |")
    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.write_text("# LLM-AggreFact: DeepEval and the LLM judge as factual-consistency checkers\n\n"
                           "Balanced accuracy on the test split; threshold at the midpoint of each checker's range "
                           "(judge consistency >= 3 of 5, DeepEval >= 0.5), as the benchmark does; dev-tuned "
                           "thresholds for reference.\n\n" + text + "\n")
        args.md.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
