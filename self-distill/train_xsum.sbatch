#!/bin/bash
#SBATCH --job-name=xsum-kd
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=06:00:00
# Usage: sbatch train_xsum.sbatch <config.yaml> <output_dir> [dataset_dir]
#   e.g. sbatch examples/train_xsum.sbatch examples/self-distillation_xsum.yaml \
#            ../models/xsum-1_of_2-short ../datasets/xsum_kd_data_1_of_2_short
#
# output_dir overrides the one in the YAML. dataset_dir, if given, overrides the
# datasets/test_datasets paths with <dataset_dir>/train and
# <dataset_dir>/validation (the layout generate_teacher_logprobs.py writes).
# The config actually used is saved as <output_dir>/config.yaml. Relative paths
# resolve against the directory sbatch was run from.
set -euo pipefail
USAGE="usage: sbatch train_xsum.sbatch <config.yaml> <output_dir> [dataset_dir]"
CONFIG=${1:?$USAGE}
OUTPUT_DIR=${2:?$USAGE}
DATASET_DIR=${3:-}

cd "$SLURM_SUBMIT_DIR"

# #SBATCH directives can't see script arguments, so send stdout/stderr to the
# output dir from here. Anything printed before this line (e.g. a usage error)
# lands in Slurm's default slurm-<jobid>.out in the submit dir.
mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# Required in every job script. Without it torch silently falls back to CPU --
# Axolotl does not error, it just sets device: cpu and grinds.
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Hugging Face downloads (base models, the semantic scorer's encoder) live in
# the project area, not the 100 GB home quota. See the main README.
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is the CUDA compat lib on LD_LIBRARY_PATH?'"

# Axolotl's --datasets flag only takes a string, not the list of dicts the KD
# dataset needs, so write the overrides into a copy of the config instead.
RUN_CONFIG="$OUTPUT_DIR/config.yaml"
uv run python - "$CONFIG" "$RUN_CONFIG" "$OUTPUT_DIR" "$DATASET_DIR" <<'EOF'
import os, sys, yaml
src, dst, output_dir, dataset_dir = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["output_dir"] = output_dir
if dataset_dir:
    for key, split in (("datasets", "train"), ("test_datasets", "validation")):
        path = os.path.join(dataset_dir, split)
        if not os.path.isdir(path):
            sys.exit(f"{path} not found")
        for ds in cfg[key]:
            ds["path"] = path
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"output_dir: {cfg['output_dir']}")
print(f"train:      {[d['path'] for d in cfg['datasets']]}")
print(f"validation: {[d['path'] for d in cfg['test_datasets']]}")
EOF

# Preprocess separately, as Axolotl itself recommends -- it keeps the
# memory-hungry tokenisation pass out of the training process.
uv run axolotl preprocess "$RUN_CONFIG"
uv run axolotl train      "$RUN_CONFIG"
