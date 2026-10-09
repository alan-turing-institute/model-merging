#!/bin/bash
#SBATCH --job-name=samsum
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=02:00:00
#SBATCH --array=0-5
# Train the arms of the SAMSum data-scaling curve, one per array task.
#
# Usage, from train/:
#   sbatch train_samsum.sbatch                      # all six arms
#   sbatch --array=0 train_samsum.sbatch            # just the largest arm
#   sbatch train_samsum.sbatch samsum_gemma.yaml ../datasets/samsum
#
# A JOB ARRAY rather than six submissions: the arms must differ only in how many
# training rows they see, and a single script reading a single config is the
# cheapest way to guarantee that. The config actually used is saved as
# <output_dir>/config.yaml, so what each arm ran is recoverable from the arm.
#
# Two hours rather than BillSum's four: SAMSum sequences are a third as long, and
# the largest arm here is 8,000 rows at four epochs.
set -euo pipefail
CONFIG=${1:-samsum_gemma.yaml}
DATA_ROOT=${2:-../datasets/samsum}

# Largest first: it is both the longest job (so starting it first shortens the
# wall clock) and the curve's ceiling, without which no step below it can be
# read as saturation rather than as undertrained.
ARMS=(n8000 n4000 n2000 n1000 n500 n250)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
DATASET_DIR="$DATA_ROOT/$ARM"
OUTPUT_DIR="../models/gemma3-samsum-$ARM-lora"

cd "$SLURM_SUBMIT_DIR"

# #SBATCH directives can't see script arguments, so send stdout/stderr to the
# output dir from here.
mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID:-0}.out" 2>&1

echo "arm=$ARM  dataset=$DATASET_DIR  output=$OUTPUT_DIR"

# Required in every job script. Without it torch silently falls back to CPU --
# Axolotl does not error, it just sets device: cpu and grinds.
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# HF_HUB_CACHE, NOT HF_HOME. HF_HOME relocates the auth token as well as the
# cache, which silently logs you out: gemma-3-4b-it is a gated repo, so the next
# fetch 401s instead of failing closed, and on an offline compute node it simply
# cannot find the weights.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$PROJECTDIR/$USER/mm/hf-cache}

# Node-local scratch, per arm so concurrent array tasks cannot collide.
export MM_PREPARED="${TMPDIR:-/tmp}/axolotl-prepared-samsum-$ARM"
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
    echo "$DATASET_DIR/$split not found - run prepare_samsum_data.py on a login node first" >&2
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
# Print the row count the arm ACTUALLY has, so the curve's x-axis is verifiable
# from the log rather than inferred from the arm's name.
n = len(load_from_disk(os.path.join(dataset_dir, "train")))
print(f"output_dir: {cfg['output_dir']}")
print(f"train rows on disk: {n}")
EOF

# Preprocess separately, as Axolotl itself recommends -- it keeps the
# memory-hungry tokenisation pass out of the training process.
uv run axolotl preprocess "$RUN_CONFIG"
uv run axolotl train      "$RUN_CONFIG"

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
