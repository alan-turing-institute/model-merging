#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
# Train the arms of a domain-split headroom screen, one per array task.
# Generic over tasks: the arm list is the ARMS file the task's prepare script
# writes, so the array size is set at submission (run_screen.sh does this).
#
#   sbatch --array=0-3 --job-name=clinc150 train_screen.sbatch clinc150
#   sbatch --array=3 train_screen.sbatch mt_domain 2     # arm 3, seed 2
#   sbatch --array=0 train_screen.sbatch billsum_split "" google/gemma-3-1b-it
set -euo pipefail
TASK=${1:?usage: train_screen.sbatch <task>}
DATA_ROOT=../datasets/$TASK
cd "$SLURM_SUBMIT_DIR"
mapfile -t ARMS < "$DATA_ROOT/ARMS"
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
DATASET_DIR="$DATA_ROOT/$ARM"
# SEED (optional second argument) reruns an arm on the same data with a
# different training seed, to measure the run-to-run spread the screens'
# "x seed spread" multiples assume. The adapter is renamed <arm>_s<seed> so it
# cannot overwrite the original. An argument, not --export=ALL: exporting the
# login environment carried its TMPDIR to compute nodes, where it does not
# exist, and every run failed in datasets.
SEED=${2:-}
export SEED
# BASE_MODEL (optional third argument) trains the same arm on a different base
# model, e.g. google/gemma-3-1b-it, with every other setting from the task's
# config unchanged. The adapter gets the model's size as a suffix (full_1b) so
# it cannot overwrite the 4B arm.
BASE_MODEL=${3:-}
export BASE_MODEL
SIZE=${BASE_MODEL:+_$(grep -oiE '[0-9]+b' <<<"${BASE_MODEL##*/}" | head -1 | tr 'B' 'b')}
OUTPUT_DIR="../models/gemma3-$TASK-$ARM${SEED:+_s$SEED}${SIZE}-lora"
mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/$TASK-${SLURM_JOB_ID:-0}.out" 2>&1
echo "task=$TASK arm=$ARM dataset=$DATASET_DIR output=$OUTPUT_DIR"

source ./cuda_env.sh
export MM_PREPARED="${TMPDIR:-/tmp}/axolotl-prepared-$TASK-$ARM${SIZE}"
mkdir -p "$MM_PREPARED"
uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU - see cuda_env.sh'"

RUN_CONFIG="$OUTPUT_DIR/config.yaml"
uv run python - "${TASK}_gemma.yaml" "$RUN_CONFIG" "$OUTPUT_DIR" "$DATASET_DIR" <<'EOF'
import math, os, sys, yaml
from datasets import load_from_disk
src, dst, output_dir, dataset_dir = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg["output_dir"] = output_dir
if os.environ.get("SEED"): cfg["seed"] = int(os.environ["SEED"])
if os.environ.get("BASE_MODEL"): cfg["base_model"] = os.environ["BASE_MODEL"]
for key, split in (("datasets", "train"), ("test_datasets", "validation")):
    for ds in cfg[key]:
        ds["path"] = os.path.join(dataset_dir, split)
rows = len(load_from_disk(os.path.join(dataset_dir, "train")))
# Our key, not axolotl's (popped before axolotl sees the config): raise the
# epoch count until a small arm gets this many optimiser steps.
min_steps = cfg.pop("mm_min_optimiser_steps", None)
if min_steps:
    batch = cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"]
    cfg["num_epochs"] = max(cfg["num_epochs"], math.ceil(min_steps * batch / rows))
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"train rows on disk: {rows}; epochs: {cfg['num_epochs']}")
EOF
uv run axolotl preprocess "$RUN_CONFIG"
uv run axolotl train      "$RUN_CONFIG"
