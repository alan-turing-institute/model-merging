#!/bin/bash
#SBATCH --job-name=crime-train
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=06:00:00
# Usage, from train/:
#   sbatch train_crime.sbatch <config.yaml> <dataset_dir> <output_dir>
#   e.g. sbatch train_crime.sbatch crime_gemma.yaml \
#            ../datasets/crime_dataset_1_of_2 ../models/gemma3-4b-crime-1-of-2-lora
#
# dataset_dir must hold train/ and validation/ (the layout prepare_data_crime.py
# writes); they replace the datasets/test_datasets paths in the config, and
# output_dir replaces its output_dir. dataset_num_proc is capped at 16 if the
# config does not set it. Everything else in the config is used
# unchanged. The config actually used is saved as <output_dir>/config.yaml.
# Relative paths resolve against train/.
set -euo pipefail
USAGE="usage: sbatch train_crime.sbatch <config.yaml> <dataset_dir> <output_dir>"
CONFIG=${1:?$USAGE}
DATASET_DIR=${2:?$USAGE}
OUTPUT_DIR=${3:?$USAGE}

# Submit from train/: uv needs its pyproject.toml to find the environment.
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml && -f prepare_data_crime.py ]] || { echo "submit from train/, not $PWD" >&2; exit 1; }
[[ -f $CONFIG ]] || { echo "$CONFIG not found" >&2; exit 1; }

# #SBATCH directives can't see script arguments, so send stdout/stderr to the
# output dir from here. Anything printed before this line (e.g. a usage error)
# lands in Slurm's default slurm-<jobid>.out in the submit dir.
mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

# The nodes' driver (580.173) now supports the cu130 wheels natively, and the
# older CUDA 13.0 forward-compat libcuda (580.65) breaks it with error 803. sbatch
# copies the submitting shell's LD_LIBRARY_PATH, so strip the compat dir from it.
LD_LIBRARY_PATH=$(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH

# Hugging Face downloads (the base model) live in the project area, not the
# 100 GB home quota. See the main README.
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is a CUDA compat lib on LD_LIBRARY_PATH?'"

# Write a copy of the config with the dataset and output paths changed (and
# dataset_num_proc capped, below).
RUN_CONFIG="$OUTPUT_DIR/config.yaml"
uv run python - "$CONFIG" "$RUN_CONFIG" "$DATASET_DIR" "$OUTPUT_DIR" <<'EOF'
import os, sys, yaml
src, dst, dataset_dir, output_dir = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["output_dir"] = output_dir
for key, split in (("datasets", "train"), ("test_datasets", "validation")):
    path = os.path.join(dataset_dir, split)
    if not os.path.isdir(path):
        sys.exit(f"{path} not found")
    for ds in cfg[key]:
        ds["path"] = path
# Axolotl defaults dataset_num_proc to the node's 288 cores, not the job's
# allocation, and each worker holds its own tokenizer: 288 of them ran the job
# out of memory. A few thousand examples need only a handful.
cfg.setdefault("dataset_num_proc", min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", 16))))
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"output_dir: {cfg['output_dir']}")
print(f"train:      {[d['path'] for d in cfg['datasets']]}")
print(f"validation: {[d['path'] for d in cfg['test_datasets']]}")
print(f"dataset_num_proc: {cfg['dataset_num_proc']}")
EOF

# Preprocess separately, as Axolotl itself recommends -- it keeps the
# memory-hungry tokenisation pass out of the training process.
uv run axolotl preprocess "$RUN_CONFIG"
uv run axolotl train      "$RUN_CONFIG"
