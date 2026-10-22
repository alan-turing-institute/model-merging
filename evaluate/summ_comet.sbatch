#!/bin/bash
#SBATCH --job-name=summ-comet
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# COMET-22 on SummEval's 1,700 summaries (16 rated systems and the reference),
# for summeval_meta.py's correlation with the experts. From evaluate/:
#   sbatch summ_comet.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../train/cuda_env.sh
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECTDIR/$USER/mm/hf-cache}
METRICS_VENV=${METRICS_VENV:-$PROJECTDIR/$USER/mm/venvs/metrics}
IN=../results/summ_comet_inputs/summeval
(cd ../train && uv run --no-sync python - "$IN" <<'PY'
import json, sys
from pathlib import Path
from datasets import load_from_disk
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = load_from_disk("../datasets/summeval/pairs")
by = {}
for r in rows:
    by.setdefault(r["system"], []).append({"id": r["id"], "src": r["document"], "mt": r["summary"], "ref": r["reference"]})
for s, rs in by.items():
    (out / f"summeval-{s}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rs))
print(f"exported {sum(map(len, by.values()))} summaries for {len(by)} systems")
PY
)
"$METRICS_VENV/bin/python" summ_comet.py "$IN" ../results/summ_metrics_summeval/summeval
