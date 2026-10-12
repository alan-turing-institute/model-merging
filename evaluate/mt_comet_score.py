"""COMET-22 scores for the exported translation runs, with paired bootstrap
intervals on each base-to-trained gain. Runs in the metrics environment
(unbabel-comet), not this one:

    $METRICS_VENV/bin/python mt_comet_score.py ../results/mt_comet_inputs

Unbabel/wmt22-comet-da, reference-based (source, translation, reference). The
system score is the mean of segment scores, as COMET reports it, so the gain
interval is a paired bootstrap over segments - the same pairing as the chrF
and BLEU intervals. Needs the model in HF_HUB_CACHE; compute nodes are offline.
"""

import json
import sys
from pathlib import Path

import numpy as np
from comet import download_model, load_from_checkpoint

MODEL = "Unbabel/wmt22-comet-da"
RESAMPLES = 2000


def main():
    in_dir = Path(sys.argv[1])
    model = load_from_checkpoint(download_model(MODEL))
    scores = {}
    for path in sorted(in_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        out = model.predict([{"src": r["src"], "mt": r["mt"], "ref": r["ref"]} for r in rows],
                            batch_size=64, gpus=1, progress_bar=False)
        scores[path.stem] = dict(zip((r["id"] for r in rows), out.scores))
        print(f"{path.stem:24} COMET {100 * out.system_score:6.2f}  (n={len(rows)})")

    report = {"model": MODEL, "runs": {k: 100 * float(np.mean(list(v.values()))) for k, v in scores.items()}, "gains": {}}
    rng = np.random.default_rng(0)
    for task in ("mt_domain", "mt_zh"):
        for size in ("4b", "1b"):
            b, t = scores.get(f"{task}_{size}_base"), scores.get(f"{task}_{size}_trained")
            if not (b and t):
                continue
            ids = sorted(set(b) & set(t), key=str)
            diff = 100 * (np.array([t[i] for i in ids]) - np.array([b[i] for i in ids]))
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(RESAMPLES)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            report["gains"][f"{task}_{size}"] = {"gain": float(diff.mean()), "ci": [float(lo), float(hi)], "n": len(ids)}
            print(f"{task} {size} gain  COMET {diff.mean():+6.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    (in_dir.parent / "mt-comet.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {in_dir.parent / 'mt-comet.json'}")


if __name__ == "__main__":
    main()
