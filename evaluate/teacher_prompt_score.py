"""Teacher-prompt comparison for XSum self-distillation: scoring.

Scores each prompt's teacher summaries (self-distill/teacher_prompt_sweep.py)
against the validation references: ROUGE-1/2/L, embedding similarity and
entity support (semantic_metrics.py), length, and the share that are a single
sentence. Each prompt is compared with P0_current by paired bootstrap.

    python teacher_prompt_score.py <dir of .jsonl>
"""

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rouge_score import rouge_scorer  # noqa: E402

from scaling_curve import paired_bootstrap  # noqa: E402
from semantic_metrics import score_one  # noqa: E402

ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
SENTENCE_END = re.compile(r"[.!?][\"'”’)]?\s+\S")


def score(rows):
    out = {k: {} for k in ("rouge1", "rouge2", "rougeL", "semantic", "entity_support", "words", "one_sentence")}
    for r in rows:
        p, i = r["prediction"], r["id"]
        s = ROUGE.score(r["reference"], p)
        for k in ("rouge1", "rouge2", "rougeL"):
            out[k][i] = s[k].fmeasure
        sem = score_one(p, r["reference"], r["document"])
        out["semantic"][i] = sem.get("semantic", 0.0)
        if sem.get("entity_support") is not None:
            out["entity_support"][i] = sem["entity_support"]
        out["words"][i] = len(p.split())
        out["one_sentence"][i] = float(bool(p) and not SENTENCE_END.search(p.strip()))
    return out


def main():
    folder = Path(sys.argv[1])
    results = {f.stem: score([json.loads(l) for l in f.read_text().splitlines()]) for f in sorted(folder.glob("*.jsonl"))}
    cols = ["rouge1", "rouge2", "rougeL", "semantic", "entity_support", "words", "one_sentence"]
    print(f"{'prompt':22}" + "".join(f"{c:>15}" for c in cols))
    for name, m in results.items():
        print(f"{name:22}" + "".join(f"{statistics.fmean(m[c].values()):15.3f}" for c in cols))
    base = results.get("P0_current")
    if base:
        print("\npaired difference from P0_current, 95% CI:")
        for name, m in results.items():
            if name == "P0_current":
                continue
            parts = []
            for c in ("rouge1", "entity_support"):
                g, lo, hi, n = paired_bootstrap(base[c], m[c])
                parts.append(f"{c} {g:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"  {name:22} " + "   ".join(parts))
    (folder.parent / "teacher-prompt-sweep.json").write_text(json.dumps(
        {n: {c: statistics.fmean(m[c].values()) for c in cols} for n, m in results.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
