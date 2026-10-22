"""COMET-22 on summaries, to test whether a translation metric tracks summary
quality at all. Runs in the metrics environment (unbabel-comet):

    $METRICS_VENV/bin/python summ_comet.py <inputs dir> <records dir>

Each <label>.jsonl in the inputs holds {id, src, mt, ref} - source article,
summary, reference - and gets <records dir>/<label>.comet-wmt22.jsonl of
{id, comet} (x 100), the per-sample format summeval_meta.py reads.

Caveats, which the SummEval correlations are there to measure: wmt22-comet-da
was trained on human ratings of translations, not summaries, and its XLM-R
encoder reads at most 512 tokens of each input, so most articles are cut.
"""

import json
import sys
from pathlib import Path

from comet import download_model, load_from_checkpoint

MODEL = "Unbabel/wmt22-comet-da"


def main():
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_from_checkpoint(download_model(MODEL))
    for path in sorted(in_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        # num_workers=0: with article-length sources the DataLoader workers'
        # shared-memory batches died with a bus error (SIGBUS) on the GH200 nodes.
        out = model.predict([{"src": r["src"], "mt": r["mt"], "ref": r["ref"]} for r in rows],
                            batch_size=32, gpus=1, progress_bar=False, num_workers=0)
        (out_dir / f"{path.stem}.comet-wmt22.jsonl").write_text(
            "".join(json.dumps({"id": r["id"], "comet": 100 * float(s)}) + "\n" for r, s in zip(rows, out.scores)))
        print(f"{path.stem:24} COMET {100 * out.system_score:6.2f}  (n={len(rows)})", flush=True)


if __name__ == "__main__":
    main()
