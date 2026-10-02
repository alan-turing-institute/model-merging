#!/bin/bash
#SBATCH --job-name=billsum
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
#SBATCH --array=0-3
# Train the four arms of the BillSum data-scaling curve, one per array task.
#
# Usage, from train/:
#   sbatch train_billsum.sbatch                       # all four arms
#   sbatch --array=0 train_billsum.sbatch             # just the full arm
#   sbatch train_billsum.sbatch billsum_gemma.yaml ../datasets/billsum
#
# A JOB ARRAY rather than four submissions: the arms must differ only in how many
# training rows they see, and a single script reading a single config is the
# cheapest way to guarantee that. The config actually used is saved as
# <output_dir>/config.yaml, so what each arm ran is recoverable from the arm.
#
# Arms run concurrently when nodes are free, so the wall clock is the full arm
# (~75 min) rather than the sum (~2.5 GPU-hours).
set -euo pipefail
CONFIG=${1:-billsum_gemma.yaml}
DATA_ROOT=${2:-../datasets/billsum}

# Index order is largest first so that if the allocation is tight and later array
# tasks are cancelled, what survives is the end of the curve that is hardest to
# infer from the others.
ARMS=(full half quarter eighth)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
DATASET_DIR="$DATA_ROOT/$ARM"
OUTPUT_DIR="../models/gemma3-billsum-$ARM-lora"

cd "$SLURM_SUBMIT_DIR"

# #SBATCH directives can't see script arguments, so send stdout/stderr to the
# output dir from here. Anything printed before this line lands in Slurm's
# default slurm-<jobid>_<taskid>.out in the submit dir.
mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID:-0}.out" 2>&1

echo "arm=$ARM  dataset=$DATASET_DIR  output=$OUTPUT_DIR"

# Required in every job script. Without it torch silently falls back to CPU --
# Axolotl does not error, it just sets device: cpu and grinds.
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Hugging Face downloads (base models, the semantic scorer's encoder) live in
# the project area, not the 100 GB home quota. See the main README.
#
# HF_HUB_CACHE, NOT HF_HOME. HF_HOME relocates the auth token as well as the
# cache, which silently logs you out: gemma-3-4b-it is a gated repo, so the next
# fetch 401s instead of failing closed, and on an offline compute node it simply
# cannot find the weights. The token stays at ~/.cache/huggingface/token where
# `hf auth login` put it. Checked on this machine: the models actually live in
# $PROJECTDIR/$USER/mm/hf-cache (39 GB, gemma plus the scorer's encoders), and
# $PROJECTDIR/$USER/hf_cache holds datasets only.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECTDIR/$USER/mm/hf-cache}

# Axolotl's tokenised-dataset cache reached 17 GB on the XSum runs and took the
# home filesystem to its quota, which stops every job on the machine. Node-local
# scratch, per arm so concurrent array tasks cannot collide on it.
export MM_PREPARED="${TMPDIR:-/tmp}/axolotl-prepared-billsum-$ARM"
mkdir -p "$MM_PREPARED"

# Air-gapped compute nodes: axolotl's telemetry retries against posthog.com
# regardless, and on the XSum ctx arm that turned a 20-minute step into five
# hours of retries before the wall clock killed it.
export DO_NOT_TRACK=1
export AXOLOTL_DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1

uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is the CUDA compat lib on LD_LIBRARY_PATH?'"

for split in train validation; do
  [[ -d "$DATASET_DIR/$split" ]] || {
    echo "$DATASET_DIR/$split not found - run prepare_billsum_data.py on a login node first" >&2
    exit 1
  }
done

# Axolotl's --datasets flag only takes a string, not the list of dicts the config
# needs, so write the overrides into a copy of the config instead.
RUN_CONFIG="$OUTPUT_DIR/config.yaml"
uv run python - "$CONFIG" "$RUN_CONFIG" "$OUTPUT_DIR" "$DATASET_DIR" <<'EOF'
import os, sys, yaml
from datasets import load_from_disk
src, dst, output_dir, dataset_dir = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["output_dir"] = output_dir
for key, split in (("datasets", "train"), ("test_datasets", "validation")):
    path = os.path.join(dataset_dir, split)
    for ds in cfg[key]:
        ds["path"] = path
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
# Print the row count the arm ACTUALLY has. Axolotl's default
# excess_length_strategy drops examples longer than sequence_len (the config
# sets `raise` so that fails the job instead), so an arm could otherwise contain
# fewer rows than its name claims, and the scaling curve's x-axis would be wrong.
n = len(load_from_disk(os.path.join(dataset_dir, "train")))
print(f"output_dir: {cfg['output_dir']}")
print(f"train rows on disk: {n}")
EOF

# Preprocess separately, as Axolotl itself recommends -- it keeps the
# memory-hungry tokenisation pass out of the training process.
uv run axolotl preprocess "$RUN_CONFIG"
uv run axolotl train      "$RUN_CONFIG"

# How many examples survived tokenisation, against how many went in. A gap here
# means sequence_len is too small for the word cap in prepare_billsum_data.py,
# and the arm is not the size its name says.
uv run python - "$MM_PREPARED" <<'EOF'
import sys
from pathlib import Path
from datasets import load_from_disk
for d in sorted(Path(sys.argv[1]).glob("*")):
    try:
        print(f"tokenised {d.name}: {len(load_from_disk(str(d)))} rows")
    except Exception:
        pass
EOF
