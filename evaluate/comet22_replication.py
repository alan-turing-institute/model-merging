"""Replicate COMET-22's published correlations, to confirm our COMET set-up.

    cd train && uv run --no-sync python ../evaluate/comet22_replication.py export ../results/comet22_replication
    $METRICS_VENV/bin/python comet22_replication.py score ../results/comet22_replication

Paper: Rei et al. (2022), COMET-22: Unbabel-IST 2022 Submission for the Metrics
Shared Task, WMT 2022, pp. 578-585. Its "COMET-22" row is an ensemble (a DA
estimator plus an MQM sequence tagger, weighted per language pair), which was
not released as one checkpoint. The public Unbabel/wmt22-comet-da - the model
this project uses - is the ensemble's DA Estimator: XLM-R trained on WMT17-20
direct assessments. So the target is the paper's "DA Estimator" row, with the
lexical baselines as a control on the data and the correlation code:

  Table 1  segment-level Spearman rho / Kendall tau over all segments of a pair
  Table 2  system-level pairwise accuracy (Kocmi et al., 2021): the share of
           system pairs a metric orders as MQM does

on the WMT21 News MQM annotations (Freitag et al., 2021): zh-en 9,750
segments / 15 systems, en-de 8,959 / 17, en-ru 8,432 / 16. Data:
RicardoRei/wmt-mqm-human-evaluation (year 2021, domain news). Our run uses the
same calls as everywhere else in this repo: load_from_checkpoint(download_model),
predict(batch_size, gpus), scores x 100.
"""

import json
import sys
from pathlib import Path

LPS = ("zh-en", "en-de", "en-ru")
PAPER = {  # Table 1 (rho, tau) and Table 2 (accuracy) of the paper
    "DA Estimator (= wmt22-comet-da)": {"seg": {"zh-en": (0.495, 0.362), "en-de": (0.289, 0.221), "en-ru": (0.369, 0.285)},
                                        "sys": {"zh-en": 50.47, "en-de": 71.32, "en-ru": 66.66}},
    "COMET-22 (ensemble, not released)": {"seg": {"zh-en": (0.537, 0.395), "en-de": (0.366, 0.281), "en-ru": (0.407, 0.315)},
                                          "sys": {"zh-en": 64.76, "en-de": 86.76, "en-ru": 70.83}},
    "BLEU": {"seg": {"zh-en": (0.215, 0.153), "en-de": (0.086, 0.065), "en-ru": (0.123, 0.094)},
             "sys": {"zh-en": 45.71, "en-de": 66.91, "en-ru": 46.66}},
    "chrF": {"seg": {"zh-en": (0.116, 0.088), "en-de": (0.116, 0.088), "en-ru": (0.213, 0.165)},
             "sys": {"zh-en": 43.81, "en-de": 65.44, "en-ru": 55.00}},
}
COUNTS = {"zh-en": (9750, 15), "en-de": (8959, 17), "en-ru": (8432, 16)}


def export(out):
    from datasets import load_dataset
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train")
    ds = ds.filter(lambda r: r["year"] == 2021 and r["domain"] == "news" and r["lp"] in LPS)
    for lp in LPS:
        rows = [r for r in ds if r["lp"] == lp]
        (out / f"{lp}.jsonl").write_text("".join(json.dumps(
            {k: r[k] for k in ("src", "mt", "ref", "score", "system")}, ensure_ascii=False) + "\n" for r in rows))
        n_sys = len({r["system"] for r in rows})
        print(f"{lp}: {len(rows)} segments, {n_sys} systems (paper: {COUNTS[lp][0]}, {COUNTS[lp][1]})")


def pairwise_accuracy(metric, human):
    systems = sorted(metric)
    agree = total = 0
    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            total += 1
            agree += (metric[a] - metric[b]) * (human[a] - human[b]) > 0
    return 100 * agree / total


def score(in_dir):
    import numpy as np
    import sacrebleu
    from comet import download_model, load_from_checkpoint
    from scipy.stats import kendalltau, spearmanr

    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    ours = {"wmt22-comet-da (ours)": {"seg": {}, "sys": {}}, "BLEU (ours)": {"seg": {}, "sys": {}},
            "chrF (ours)": {"seg": {}, "sys": {}}}
    counts = {}
    for lp in LPS:
        rows = [json.loads(l) for l in (in_dir / f"{lp}.jsonl").read_text().splitlines()]
        for r in rows:  # a few MQM rows have no translation; score them as empty, as the counts include them
            for k in ("src", "mt", "ref"):
                r[k] = r[k] or ""
        counts[lp] = (len(rows), len({r["system"] for r in rows}))
        human = np.array([r["score"] for r in rows])
        comet = np.array(model.predict([{k: r[k] for k in ("src", "mt", "ref")} for r in rows],
                                       batch_size=64, gpus=1, progress_bar=False).scores) * 100
        bleu = np.array([sacrebleu.sentence_bleu(r["mt"], [r["ref"]]).score for r in rows])
        chrf = np.array([sacrebleu.sentence_chrf(r["mt"], [r["ref"]]).score for r in rows])
        for name, m in (("wmt22-comet-da (ours)", comet), ("BLEU (ours)", bleu), ("chrF (ours)", chrf)):
            ours[name]["seg"][lp] = (float(spearmanr(m, human).statistic), float(kendalltau(m, human).statistic))
            sysm, sysh = {}, {}
            for r, x, h in zip(rows, m, human):
                sysm.setdefault(r["system"], []).append(x)
                sysh.setdefault(r["system"], []).append(h)
            ours[name]["sys"][lp] = pairwise_accuracy({s: np.mean(v) for s, v in sysm.items()},
                                                      {s: np.mean(v) for s, v in sysh.items()})
        print(f"{lp} done", flush=True)

    md = ["# COMET-22 replication (Rei et al., 2022, WMT)", "",
          "WMT21 News MQM. Segments / systems here: " + ", ".join(f"{lp} {counts[lp][0]} / {counts[lp][1]}" for lp in LPS)
          + "; paper: " + ", ".join(f"{lp} {COUNTS[lp][0]} / {COUNTS[lp][1]}" for lp in LPS) + ".", "",
          "| metric | " + " | ".join(f"{lp} rho / tau" for lp in LPS) + " | avg rho / tau | "
          + " | ".join(f"{lp} sys acc" for lp in LPS) + " | avg acc |", "|---|" + "---|" * 8]
    for name, r in list(PAPER.items()) + list(ours.items()):
        seg = [r["seg"][lp] for lp in LPS]
        acc = [r["sys"][lp] for lp in LPS]
        md.append(f"| {name}{'' if '(ours)' in name else ' (paper)'} | "
                  + " | ".join(f"{a:.3f} / {b:.3f}" for a, b in seg)
                  + f" | {np.mean([a for a, _ in seg]):.3f} / {np.mean([b for _, b in seg]):.3f} | "
                  + " | ".join(f"{a:.2f}" for a in acc) + f" | {np.mean(acc):.2f} |")
    text = "\n".join(md)
    print(text)
    (in_dir.parent / "comet22-replication.md").write_text(text + "\n")
    (in_dir.parent / "comet22-replication.json").write_text(json.dumps({"ours": ours, "paper": PAPER, "counts": counts}, indent=1) + "\n")


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]](Path(sys.argv[2]))
