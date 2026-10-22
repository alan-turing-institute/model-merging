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
# The tokenised dataset is cached per run in <output_dir>/prepared.
# The config actually used is saved as <output_dir>/config.yaml. Relative paths
# resolve against the directory sbatch was run from, which must be self-distill/.
# Once training succeeds, <output_dir>/provenance.json records what made the
# model (see ../provenance.py).
set -euo pipefail
USAGE="usage: sbatch train_xsum.sbatch <config.yaml> <output_dir> [dataset_dir]"
CONFIG=${1:?$USAGE}
OUTPUT_DIR=${2:?$USAGE}
DATASET_DIR=${3:-}

cd "$SLURM_SUBMIT_DIR"
[[ -f train_xsum.sbatch ]] || { echo "submit from self-distill/, not $PWD" >&2; exit 1; }

# #SBATCH directives can't see script arguments, so send stdout/stderr to the
# output dir from here. Anything printed before this line (e.g. a usage error)
# lands in Slurm's default slurm-<jobid>.out in the submit dir.
mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1
# Nothing more goes to slurm-<jobid>.out, so on exit delete it if it is empty.
# Kept if it holds an early error or a message from Slurm (e.g. out of memory).
trap '[[ -s slurm-$SLURM_JOB_ID.out ]] || rm -f "slurm-$SLURM_JOB_ID.out"' EXIT

# The nodes' driver (580.173) now supports the cu130 wheels natively, and the
# older CUDA 13.0 forward-compat libcuda (580.65) breaks it with error 803. sbatch
# copies the submitting shell's LD_LIBRARY_PATH, so strip the compat dir from it.
LD_LIBRARY_PATH=$(tr ':' '
' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '/compat$' || true; } | paste -sd: -)
export LD_LIBRARY_PATH

# Hugging Face downloads (base models, the semantic scorer's encoder) live in
# the project area, not the 100 GB home quota. See the main README.
export HF_HOME=${HF_HOME:-$PROJECTDIR/$USER/hf_cache}

uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is a CUDA compat lib on LD_LIBRARY_PATH?'"

# Axolotl's --datasets flag only takes a string, not the list of dicts the KD
# dataset needs, so write the overrides into a copy of the config instead.
RUN_CONFIG="$OUTPUT_DIR/config.yaml"
uv run python - "$CONFIG" "$RUN_CONFIG" "$OUTPUT_DIR" "$DATASET_DIR" <<'EOF'
import os, sys, yaml
src, dst, output_dir, dataset_dir = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["output_dir"] = output_dir
# A per-run cache: Axolotl's hash ignores the base model, so concurrent runs on
# the same dataset sharing one cache overwrite each other's mmapped arrow files
# (SIGBUS in the dataloader).
cfg["dataset_prepared_path"] = os.path.join(output_dir, "prepared")
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
print(f"prepared:   {cfg['dataset_prepared_path']}")
print(f"train:      {[d['path'] for d in cfg['datasets']]}")
print(f"validation: {[d['path'] for d in cfg['test_datasets']]}")
EOF

# Preprocess separately, as Axolotl itself recommends -- it keeps the
# memory-hungry tokenisation pass out of the training process.
uv run axolotl preprocess "$RUN_CONFIG"
uv run axolotl train      "$RUN_CONFIG"

uv run python ../provenance.py model "$OUTPUT_DIR" --script train_xsum.sbatch
