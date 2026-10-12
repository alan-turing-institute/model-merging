"""Corpus BLEU for the translation screens, from the stored .eval logs.

    uv run python mt_bleu.py logs --task mt_domain
    uv run python mt_bleu.py logs --task mt_zh

The tasks score sentence-level chrF/chrF++ inline, which averages and pairs
cleanly. BLEU is different: the reported number is CORPUS BLEU - n-gram counts
summed over all sentences, then combined once - and an average of sentence
BLEU is not it. So it is computed here, after the fact, from each sample's
output and reference; nothing is re-generated.

Tokenisation is sacreBLEU's: 13a for German, `zh` for Chinese (BLEU on
unsegmented Chinese is meaningless without it).

The interval on a gain is a paired bootstrap over sentences: resample sentence
indices, recompute both corpus scores from the resampled n-gram statistics,
take the difference - the standard significance test for MT, and the same
pairing as the chrF intervals. 1,000 resamples, fixed seed.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from inspect_ai.log import list_eval_logs, read_eval_log
from sacrebleu.metrics import BLEU

TOKENIZE = {"mt_domain": "13a", "mt_zh": "zh"}
RESAMPLES = 1000


def label(log, task):
    """('4B'|'1B', 'base'|'trained'|<adapter name>) for one log."""
    model = log.eval.model
    size = "1B" if "gemma-3-1b" in model else "4B" if "gemma-3-4b" in model else model
    adapter = str((log.eval.model_args or {}).get("adapter_path", "")).rstrip("/")
    if not adapter:
        return size, "base"
    name = Path(adapter).name
    full = re.fullmatch(rf"gemma3-{task}-full(_\d+b)?-lora", name)
    return size, "trained" if full else name


def outputs(log):
    """{sample id: (hypothesis, reference)}."""
    return {s.id: (s.output.completion.strip() if s.output else "", str(s.target))
            for s in log.samples or []}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", nargs="?", default="logs", type=Path)
    parser.add_argument("--task", required=True, choices=sorted(TOKENIZE))
    args = parser.parse_args()
    bleu = BLEU(tokenize=TOKENIZE[args.task])

    logs = {}
    for info in list_eval_logs(str(args.log_dir)):
        log = read_eval_log(info)
        if log.status != "success" or log.eval.task.split("/")[-1] != args.task:
            continue
        key = label(log, args.task)
        if key not in logs or log.eval.created > logs[key].eval.created:
            logs[key] = log

    print(f"{args.task}: corpus BLEU, tokenize={TOKENIZE[args.task]}")
    for size in ("4B", "1B"):
        base, trained = logs.get((size, "base")), logs.get((size, "trained"))
        for kind, log in (("base", base), ("trained", trained)):
            if log:
                h, r = zip(*outputs(log).values())
                print(f"  {size} {kind:8} BLEU {bleu.corpus_score(list(h), [list(r)]).score:6.2f}  (n={len(h)})")
        if not (base and trained):
            continue
        ob, ot = outputs(base), outputs(trained)
        ids = sorted(set(ob) & set(ot))
        # Per-sentence sufficient statistics, so each resample is a sum, not a re-score.
        sb = np.array(bleu._extract_corpus_statistics([ob[i][0] for i in ids], [[ob[i][1] for i in ids]]))
        st = np.array(bleu._extract_corpus_statistics([ot[i][0] for i in ids], [[ot[i][1] for i in ids]]))
        score = lambda stats: bleu._compute_score_from_stats(list(stats)).score
        point = score(st.sum(0)) - score(sb.sum(0))
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(RESAMPLES):
            pick = rng.integers(0, len(ids), len(ids))
            diffs.append(score(st[pick].sum(0)) - score(sb[pick].sum(0)))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  {size} gain     BLEU {point:+6.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (paired bootstrap, n={len(ids)})")


if __name__ == "__main__":
    main()
