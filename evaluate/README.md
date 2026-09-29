# Readme for evaluating trained/merged models

`evaluate.py` runs a model as a classifier against a held-out test set and reports accuracy, precision, recall, F1, and a confusion matrix.

```bash
uv run python evaluate.py --model <model> [--adapter <adapter>] --output results/<name>.json
```

`--model` and `--adapter` each accept:

- a local path (e.g. `../models/full_model1`)
- a Hugging Face hub id (e.g. `google/gemma-3-4b-it`)
- an Azure ML registered model reference, `azureml:<name>:<version>` (e.g. `azureml:gemma3-crime-full:1`)

NB models stored in the Azure blob storage must be downloaded before they can be used.  If you pass an `azureml:` reference, the script downloads it automatically into `evaluate/.azureml_models/` (skipped on later runs if already cached there) before loading it; nothing needs to be downloaded by hand first.

Example, evaluating a registered adapter against the base model:

```bash
uv run python evaluate.py \
  --model google/gemma-3-4b-it \
  --adapter azureml:gemma3-crime-full-lora:1 \
  --output results/gemma3-crime-full-lora.json
```

`--output` writes the metrics plus per-example predictions to a JSON file — run this once per model variant and compare the resulting files (e.g. with `pandas.json_normalize`) rather than reading metrics off the terminal.

## Running on Isambard-AI

Isambard-AI's GPU driver (565.57.01) only supports up to CUDA 12.7, but the torch that `uv sync` installs here is built for CUDA 13.0. To let it see the GPU, put the CUDA 13.0 [forward-compatibility driver](https://docs.isambard.ac.uk/user-documentation/guides/gpus_and_cuda/#cuda-forward-compatibility) from the shared HPC SDK ahead of the system one:

```bash
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

This must be set in **every** shell and **every** Slurm job script, not just `.bashrc`. Without it nothing errors: `torch.cuda.is_available()` is `False` and the model quietly runs on the CPU. A quick check before a long run:

```bash
uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU: is the CUDA compat lib on LD_LIBRARY_PATH?'"
```

This path is specific to Isambard. On the Azure x86 instance, run the same check; it only fails there if that machine's driver is older than CUDA 13.0.
