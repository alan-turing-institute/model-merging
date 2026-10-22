"""COMET-22 on the EN->DE merge experiments, beside chrF++. Runs in the metrics
environment (unbabel-comet), after mt_comet_export.py --all:

    $METRICS_VENV/bin/python mt_comet_merges.py ../results/mt_comet_merge_inputs

Scores every <label>.jsonl in the directory (segment scores cached in
<dir>/scores/, so a re-run scores only new runs) and writes
../results/mt-comet-merges.json and .md with:

  data-parallel, 4B   untrained, full, IID half, each domain expert, the better
                      expert per seed and each merge method (means over seeds),
                      overall and per domain; and per method and seed the
                      recovery fraction R = (merged - better expert) /
                      (full - better expert) on COMET, with a paired bootstrap
                      95% interval over segments - merge_report.py's protocol
  data-parallel, 1B   untrained, full and the two experts (no 1B merges yet)
  task-parallel       per combination and size: untrained, the EN->DE
                      specialist, the multi-task ceiling, each merge, with
                      retention (score - base) / (specialist - base) and the
                      merge's paired interval against the specialist
  gold-target KD      the distilled EN->DE models against the full model

Unbabel/wmt22-comet-da, reference-based; system score = mean segment score x 100.
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np

MODEL = "Unbabel/wmt22-comet-da"
RESAMPLES = 2000
METHODS = ["linear", "ties", "dare_ties", "task_arithmetic"]
SEEDS = ["", "_s2", "_s3"]
COMBOS = ["clinc150-mt_domain", "mt_domain-xsumc", "clinc150-mt_domain-xsumc"]


def score_runs(in_dir):
    cache = in_dir / "scores"
    cache.mkdir(exist_ok=True)
    runs, model = {}, None
    for path in sorted(in_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        cached = cache / f"{path.stem}.json"
        if cached.exists() and len(json.loads(cached.read_text())) == len(rows):
            seg = json.loads(cached.read_text())
        else:
            if model is None:
                from comet import download_model, load_from_checkpoint
                model = load_from_checkpoint(download_model(MODEL))
            out = model.predict([{"src": r["src"], "mt": r["mt"], "ref": r["ref"]} for r in rows],
                                batch_size=64, gpus=1, progress_bar=False)
            seg = {str(r["id"]): 100 * float(s) for r, s in zip(rows, out.scores)}
            cached.write_text(json.dumps(seg))
        runs[path.stem] = {"comet": seg, "domain": {str(r["id"]): r["domain"] for r in rows},
                           "chrf": {str(r["id"]): r["chrf"] for r in rows if r.get("chrf") is not None}}
    return runs


def mean(d, ids=None, domain=None, run=None):
    vals = [v for k, v in d.items() if (ids is None or k in ids)
            and (domain is None or run["domain"].get(k) == domain)]
    return statistics.fmean(vals) if vals else None


def stats(run):
    c = run["comet"]
    return {"comet": mean(c), "comet_medical": mean(c, domain="medical", run=run),
            "comet_law": mean(c, domain="law", run=run), "chrf": mean(run["chrf"]) if run["chrf"] else None,
            "n": len(c)}


def boot_ratio(num_a, num_b, den_a, den_b, rng):
    """(mean a - mean b) / (mean c - mean b) over shared ids, with a paired bootstrap."""
    ids = sorted(set(num_a) & set(num_b) & set(den_a))
    m, b, f = (np.array([d[i] for i in ids]) for d in (num_a, num_b, den_a))
    point = (m.mean() - b.mean()) / (f.mean() - b.mean())
    boots = []
    for _ in range(RESAMPLES):
        k = rng.integers(0, len(ids), len(ids))
        den = f[k].mean() - b[k].mean()
        if den != 0:
            boots.append((m[k].mean() - b[k].mean()) / den)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), [float(lo), float(hi)]


def boot_diff(a, b, rng):
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(RESAMPLES)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), [float(lo), float(hi)]


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    in_dir = Path(sys.argv[1])
    runs = score_runs(in_dir)
    st = {k: stats(v) for k, v in runs.items()}
    rng = np.random.default_rng(0)
    report = {"model": MODEL, "runs": st, "data_parallel_4b": [], "task_parallel": [], "kd_gold": {}}
    md = ["# COMET-22 on the EN->DE merge experiments", "",
          f"`{MODEL}`, mean segment score x 100; chrF++ from the same logs.", ""]

    def row(name, labels):
        have = [st[l] for l in labels if l in st]
        if not have:
            return None
        cell = lambda k: statistics.fmean(h[k] for h in have if h[k] is not None) if any(h[k] is not None for h in have) else None
        sd = statistics.stdev(h["comet"] for h in have) if len(have) > 1 else None
        return (f"| {name} | {fmt(cell('comet'))}{' ±' + fmt(sd) if sd is not None else ''} | "
                f"{fmt(cell('comet_medical'))} | {fmt(cell('comet_law'))} | {fmt(cell('chrf'))} | {len(have)} |")

    head = ["| model | COMET | COMET medical | COMET law | chrF++ | runs |", "|---|---|---|---|---|---|"]

    # data-parallel, 4B
    full, base = "gemma3-mt_domain-full-lora", "base-gemma-3-4b-it"
    med = lambda s: f"gemma3-mt_domain-expert_medical{s}-lora"
    law = lambda s: f"gemma3-mt_domain-expert_law{s}-lora"
    merged = lambda m, s: f"mt_domain-merged-{m}{s}"
    better = []
    for s in SEEDS:
        if med(s) in st and law(s) in st:
            better.append(med(s) if st[med(s)]["comet"] >= st[law(s)]["comet"] else law(s))
    md += ["## Data-parallel, 4B (domain experts: medical vs law)", ""] + head
    for name, labels in [("untrained", [base]), ("full data", [full]), ("IID half", ["gemma3-mt_domain-iid_half-lora"]),
                         ("expert: medical", [med(s) for s in SEEDS]), ("expert: law", [law(s) for s in SEEDS]),
                         ("better expert", better)] + [(f"merged: {m}", [merged(m, s) for s in SEEDS]) for m in METHODS]:
        r = row(name, labels)
        if r:
            md.append(r)
    md += ["", "| method | seed | merged | better expert | full | R | 95% CI |", "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for s in SEEDS:
            if merged(m, s) not in runs or med(s) not in runs or law(s) not in runs or full not in runs:
                continue
            best = med(s) if st[med(s)]["comet"] >= st[law(s)]["comet"] else law(s)
            r, ci = boot_ratio(runs[merged(m, s)]["comet"], runs[best]["comet"], runs[full]["comet"],
                               runs[best]["comet"], rng)
            seed = s.strip("_s") or "1"
            report["data_parallel_4b"].append({"method": m, "seed": seed, "merged": st[merged(m, s)]["comet"],
                                               "better_expert": st[best]["comet"], "full": st[full]["comet"],
                                               "R": r, "ci": ci})
            md.append(f"| {m} | {seed} | {fmt(st[merged(m, s)]['comet'])} | {fmt(st[best]['comet'])} | "
                      f"{fmt(st[full]['comet'])} | {r:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    for m in METHODS:
        rs = [x["R"] for x in report["data_parallel_4b"] if x["method"] == m]
        if len(rs) > 1:
            md.append(f"| {m} | mean | | | | {statistics.fmean(rs):+.3f} (SD {statistics.stdev(rs):.3f}) | |")

    # data-parallel, 1B
    md += ["", "## Data-parallel, 1B (no 1B merges yet)", ""] + head
    for name, labels in [("untrained", ["base-gemma-3-1b-it"]), ("full data", ["gemma3-mt_domain-full_1b-lora"]),
                         ("expert: medical", ["gemma3-mt_domain-expert_medical_1b-lora"]),
                         ("expert: law", ["gemma3-mt_domain-expert_law_1b-lora"])]:
        r = row(name, labels)
        if r:
            md.append(r)

    # task-parallel
    md += ["", "## Task-parallel merges, scored on EN->DE", "",
           "Retention = (score - untrained) / (EN->DE specialist - untrained); the interval is merged minus specialist.", ""]
    for size in ("4b", "1b"):
        sfx, pre = ("_1b", "tp-1b-") if size == "1b" else ("", "tp-")
        b, spec = f"base-gemma-3-{size}-it", f"gemma3-mt_domain-full{sfx}-lora"
        if b not in st or spec not in st:
            continue
        for combo in COMBOS:
            ceiling = f"gemma3-tp_{combo.replace('-', '_')}-full{sfx}-lora"
            names = [(f"merged: {m}", f"{pre}{combo}-merged-{m}") for m in METHODS]
            if not any(l in st for _, l in names):
                continue
            md += [f"### {combo}, {size}", "", "| model | COMET | retention | chrF++ | merged minus specialist [95% CI] |",
                   "|---|---|---|---|---|"]
            for name, l in [("untrained", b), ("EN->DE specialist", spec), ("multi-task ceiling", ceiling)] + names:
                if l not in st:
                    continue
                ret = (st[l]["comet"] - st[b]["comet"]) / (st[spec]["comet"] - st[b]["comet"])
                diff = ""
                if name.startswith("merged"):
                    d, ci = boot_diff(runs[l]["comet"], runs[spec]["comet"], rng)
                    diff = f"{d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
                    report["task_parallel"].append({"size": size, "combo": combo, "method": name[8:], "comet": st[l]["comet"],
                                                    "retention": ret, "minus_specialist": d, "ci": ci})
                md.append(f"| {name} | {fmt(st[l]['comet'])} | {ret:.3f} | {fmt(st[l]['chrf'])} | {diff} |")
            md.append("")

    # gold-target KD
    kd = [f"gemma3-mt_domain_kdgold_ce{c}-lora" for c in (100, 90, 75, 50)]
    if any(l in st for l in kd):
        md += ["## Gold-target distillation (CE weight), 4B", ""] + head
        for l in [full] + kd:
            r = row(l.replace("gemma3-mt_domain_", "").replace("-lora", ""), [l])
            if r:
                md.append(r)
                report["kd_gold"][l] = st[l]

    out = in_dir.parent
    (out / "mt-comet-merges.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "mt-comet-merges.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
