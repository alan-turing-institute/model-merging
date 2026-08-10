"""
Compares our reproduced numbers against Table 1 of the original MergeKit
paper (Goddard et al. 2024, arXiv 2403.13257) -- same base models
(Llama-2-7B-Chat, Meditron-7B), same 4 merge methods (LERP/SLERP/TIES/
DARE-TIES), same 6 benchmarks.
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./lm-eval")

TASKS = ["medqa_4options", "medmcqa", "pubmedqa", "arc_challenge", "hellaswag", "mmlu"]
TASK_LABELS = {"medqa_4options": "USMLE", "medmcqa": "MedMCQA", "pubmedqa": "PubMedQA",
               "arc_challenge": "ARC-Challenge", "hellaswag": "HellaSwag", "mmlu": "MMLU"}
# Prefer acc_norm where the paper's own methodology (Open LLM Leaderboard
# convention) uses it; acc otherwise.
PREFERRED_METRIC = {"medqa_4options": "acc", "medmcqa": "acc", "pubmedqa": "acc",
                    "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "mmlu": "acc"}

PAPER = {
    "llama2-7b-chat":     {"medqa_4options": 35.90, "medmcqa": 35.45, "pubmedqa": 73.40, "arc_challenge": 44.20, "hellaswag": 55.40, "mmlu": 46.37},
    "meditron-7b":        {"medqa_4options": 38.40, "medmcqa": 24.07, "pubmedqa": 71.40, "arc_challenge": 40.20, "hellaswag": 54.50, "mmlu": 33.06},
    "merged-lerp":        {"medqa_4options": 39.10, "medmcqa": 36.65, "pubmedqa": 75.60, "arc_challenge": 46.76, "hellaswag": 58.66, "mmlu": 48.44},
    "merged-slerp":       {"medqa_4options": 39.20, "medmcqa": 36.91, "pubmedqa": 75.60, "arc_challenge": 46.84, "hellaswag": 58.67, "mmlu": 47.97},
    "merged-dare-ties":   {"medqa_4options": 36.37, "medmcqa": 27.56, "pubmedqa": 72.20, "arc_challenge": 42.92, "hellaswag": 54.79, "mmlu": 41.17},
    "merged-ties":        {"medqa_4options": 38.73, "medmcqa": 32.27, "pubmedqa": 75.60, "arc_challenge": 45.05, "hellaswag": 58.23, "mmlu": 45.03},
}


def load_result(model_dir, task):
    pattern = str(ROOT / model_dir / "**" / f"results_*.json")
    files = glob.glob(pattern, recursive=True)
    f = sorted(files)[-1]
    with open(f) as fh:
        data = json.load(fh)
    task_results = data["results"][task]
    metric = PREFERRED_METRIC[task]
    for key in [f"{metric},none", metric]:
        if key in task_results:
            return task_results[key] * 100
    # fall back to whatever's present
    for k, v in task_results.items():
        if k.startswith(("acc", "exact_match")):
            return v * 100
    return None


def main():
    print(f"{'Model':20s} {'Task':15s} {'Ours':>8s} {'Paper':>8s} {'Diff':>8s}")
    for model in PAPER:
        for task in TASKS:
            try:
                ours = load_result(model, task)
            except (IndexError, FileNotFoundError):
                ours = None
            paper = PAPER[model][task]
            if ours is None:
                print(f"{model:20s} {TASK_LABELS[task]:15s} {'MISSING':>8s} {paper:8.2f} {'--':>8s}")
            else:
                diff = ours - paper
                print(f"{model:20s} {TASK_LABELS[task]:15s} {ours:8.2f} {paper:8.2f} {diff:+8.2f}")
        print()


if __name__ == "__main__":
    main()
