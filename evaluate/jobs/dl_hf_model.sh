#!/bin/bash
#SBATCH --job-name=dl-hf-model
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=04:00:00
#SBATCH --output=dl-hf-model-%j.out
# Download a Hugging Face model at a pinned revision into the project area, as
# a job (the login node's memory cap kills large downloads). Gated repos use the
# token in ~/.cache/huggingface/token, which this never prints. From evaluate/:
#   sbatch jobs/dl_hf_model.sh meta-llama/Llama-3.3-70B-Instruct llama-3.3-70b-instruct
# The revision is written to <dir>/REVISION; re-submitting resumes.
set -euo pipefail
REPO=${1:?usage: dl_hf_model.sh <repo> <dir name>}
DEST=$PROJECTDIR/$USER/mm/models/${2:?usage: dl_hf_model.sh <repo> <dir name>}
cd "$SLURM_SUBMIT_DIR/../train"
uv run --no-sync python - "$REPO" "$DEST" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo, dest = sys.argv[1], Path(sys.argv[2])
rev = (dest / "REVISION").read_text().strip() if (dest / "REVISION").exists() else HfApi().model_info(repo).sha
dest.mkdir(parents=True, exist_ok=True)
(dest / "REVISION").write_text(rev + "\n")
snapshot_download(repo, revision=rev, local_dir=dest, max_workers=8,
                  allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"])
size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
print(f"{repo}@{rev[:12]} -> {dest} ({size:.0f} GB)")
PY
