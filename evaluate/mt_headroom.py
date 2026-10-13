"""Per-dataset translation headroom: untrained model vs the model trained on
that dataset alone, on that dataset's own test sentences.

    uv run python mt_headroom.py logs ../results/mt_headroom_inputs

Four datasets from the two translation screens - EMEA and JRC-Acquis (EN->DE),
UN and News Commentary (EN->ZH) - each with its single-dataset model (the
screens' expert_* arms, 4,000 training pairs) at 4B and 1B. The news model is
scored on newstest2018, the screen's news test; News Commentary has no test
split of its own here.

Reports chrF (chrF++ for German, character chrF for Chinese) as a mean of
sentence scores and corpus BLEU (sacreBLEU; 13a / zh), each with a paired
bootstrap 95% interval on the gain, and writes (source, translation, reference)
triples per dataset for COMET (mt_comet_score.py).
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from inspect_ai.log import list_eval_logs, read_eval_log
from sacrebleu.metrics import BLEU

from scaling_curve import paired_bootstrap

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {  # label -> (task, domain tag, target language, BLEU tokeniser)
    "EMEA (EN-DE medical)": ("mt_domain", "medical", "de", "13a"),
    "JRC-Acquis (EN-DE legal)": ("mt_domain", "law", "de", "13a"),
    "UN (EN-ZH)": ("mt_zh", "un", "zh", "zh"),
    "News Commentary (EN-ZH; test: newstest2018)": ("mt_zh", "news", "zh", "zh"),
}
RESAMPLES = 1000


def pick_logs(log_dir):
    """{(task, size, 'base' | expert domain): newest successful log}."""
    picked = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        task = log.eval.task.split("/")[-1]
        if log.status != "success" or task not in ("mt_domain", "mt_zh"):
            continue
        size = "1b" if "gemma-3-1b" in log.eval.model else "4b" if "gemma-3-4b" in log.eval.model else None
        adapter = Path(str((log.eval.model_args or {}).get("adapter_path", "")).rstrip("/")).name
        if not size:
            continue
        if not adapter:
            kind = "base"
        else:
            for _, (t, domain, _, _) in DATASETS.items():
                if t == task and adapter == f"gemma3-{task}-expert_{domain}{'_1b' if size == '1b' else ''}-lora":
                    kind = domain
                    break
            else:
                continue
        key = (task, size, kind)
        if key not in picked or log.eval.created > picked[key].eval.created:
            picked[key] = log
    return picked


def rows_for(log, domain):
    return [s for s in log.samples if (s.metadata or {}).get("domain") == domain]


def chrf(samples):
    out = {}
    for s in samples:
        for sc in (s.scores or {}).values():
            v = sc.value.get("chrf") if isinstance(sc.value, dict) else None
            if isinstance(v, (int, float)):
                out[s.id] = float(v)
    return out


def bleu_gain(base, trained, tokenize):
    bleu = BLEU(tokenize=tokenize)
    b = {s.id: (s.output.completion.strip() if s.output else "", str(s.target)) for s in base}
    t = {s.id: (s.output.completion.strip() if s.output else "", str(s.target)) for s in trained}
    ids = sorted(set(b) & set(t), key=str)
    sb = np.array(bleu._extract_corpus_statistics([b[i][0] for i in ids], [[b[i][1] for i in ids]]))
    st = np.array(bleu._extract_corpus_statistics([t[i][0] for i in ids], [[t[i][1] for i in ids]]))
    score = lambda stats: bleu._compute_score_from_stats(list(stats)).score
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(RESAMPLES):
        pick = rng.integers(0, len(ids), len(ids))
        diffs.append(score(st[pick].sum(0)) - score(sb[pick].sum(0)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return score(sb.sum(0)), score(st.sum(0)), score(st.sum(0)) - score(sb.sum(0)), lo, hi


def main():
    log_dir, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = pick_logs(log_dir)
    report = []
    for label, (task, domain, lang, tokenize) in DATASETS.items():
        test = load_from_disk(str(REPO_ROOT / "datasets" / task / "test"))
        for size in ("4b", "1b"):
            base_log, expert_log = logs.get((task, size, "base")), logs.get((task, size, domain))
            if not (base_log and expert_log):
                print(f"{label} {size}: missing {'base' if not base_log else 'trained'} log")
                continue
            base, trained = rows_for(base_log, domain), rows_for(expert_log, domain)
            cb, ct = chrf(base), chrf(trained)
            g, lo, hi, n = paired_bootstrap(cb, ct)
            bb, bt, bg, blo, bhi = bleu_gain(base, trained, tokenize)
            entry = {"dataset": label, "size": size, "n": n,
                     "chrf": {"base": statistics.fmean(cb.values()), "trained": statistics.fmean(ct.values()),
                              "gain": g, "ci": [lo, hi]},
                     "bleu": {"base": bb, "trained": bt, "gain": bg, "ci": [float(blo), float(bhi)]}}
            report.append(entry)
            print(f"{label:46} {size}  chrF {entry['chrf']['base']:5.1f} -> {entry['chrf']['trained']:5.1f} "
                  f"({g:+.1f} [{lo:+.1f}, {hi:+.1f}])  BLEU {bb:5.1f} -> {bt:5.1f} ({bg:+.1f} [{blo:+.1f}, {bhi:+.1f}])  n={n}")
            # Triples for COMET, the source read from the test split and the
            # reference checked against the log so nothing is mis-paired.
            for kind, samples in (("base", base), ("trained", trained)):
                lines = []
                for s in samples:
                    row = test[int(s.id)]
                    if row[lang] != str(s.target):
                        raise SystemExit(f"{task} sample {s.id}: log target does not match the test split")
                    lines.append(json.dumps({"id": s.id, "src": row["en"], "ref": row[lang],
                                             "mt": s.output.completion.strip() if s.output else ""}, ensure_ascii=False))
                (out_dir / f"{task}-{domain}_{size}_{kind}.jsonl").write_text("\n".join(lines) + "\n")
    (out_dir.parent / "mt-headroom.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_dir.parent / 'mt-headroom.json'} and COMET inputs in {out_dir}")


if __name__ == "__main__":
    main()
