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
set -euo pipefail
TASK=${1:?usage: train_screen.sbatch <task>}
DATA_ROOT=../datasets/$TASK
cd "$SLURM_SUBMIT_DIR"
mapfile -t ARMS < "$DATA_ROOT/ARMS"
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
DATASET_DIR="$DATA_ROOT/$ARM"
OUTPUT_DIR="../models/gemma3-$TASK-$ARM-lora"
mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/$TASK-${SLURM_JOB_ID:-0}.out" 2>&1
echo "task=$TASK arm=$ARM dataset=$DATASET_DIR output=$OUTPUT_DIR"

source ./cuda_env.sh
export MM_PREPARED="${TMPDIR:-/tmp}/axolotl-prepared-$TASK-$ARM"
mkdir -p "$MM_PREPARED"
uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU - see cuda_env.sh'"

RUN_CONFIG="$OUTPUT_DIR/config.yaml"
uv run python - "${TASK}_gemma.yaml" "$RUN_CONFIG" "$OUTPUT_DIR" "$DATASET_DIR" <<'EOF'
import os, sys, yaml
from datasets import load_from_disk
src, dst, output_dir, dataset_dir = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg["output_dir"] = output_dir
for key, split in (("datasets", "train"), ("test_datasets", "validation")):
    for ds in cfg[key]:
        ds["path"] = os.path.join(dataset_dir, split)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"train rows on disk: {len(load_from_disk(os.path.join(dataset_dir, 'train')))}")
EOF
uv run axolotl preprocess "$RUN_CONFIG"
uv run axolotl train      "$RUN_CONFIG"
