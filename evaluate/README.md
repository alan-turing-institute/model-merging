# Readme for evaluating trained/merged models

Evaluation runs as [Inspect AI](https://inspect.aisi.org.uk/) tasks:

- `crime_task.py` - classifies Reddit posts as `crime` / `not_crime`. Reports accuracy, precision, recall, F1, confusion-matrix counts and the share of unparseable answers.
- `xsum_task.py` - one-sentence XSum summaries. Reports ROUGE-1/2/L, embedding similarity to the reference (`semantic`), whether the names and numbers in the summary appear in the article (`entity_support`), and output length.

Both build their prompts with the same function as the script that made the training data (`train/prepare_data.py` and `self-distill/prepare_data_XSum.py`), so training and evaluation wording cannot drift apart.

Each run writes a `.eval` log to `logs/` with every sample, score and setting. Browse them with:

```bash
uv run inspect view --log-dir logs
```

and load them into pandas with `inspect_ai.analysis.samples_df("logs")` / `evals_df("logs")`.

## Running on Isambard-AI

Submit `evaluate.sbatch` from `evaluate/`:

```bash
sbatch evaluate.sbatch <crime|xsum> <model> [--adapter <dir>] [inspect eval args...]
```

`<model>` is a Hugging Face hub id or a full model saved on disk. `--adapter` puts a LoRA adapter on top of the model; this is only needed for the crime models, as the XSum models are self-distilled full models. Anything after that goes to `inspect eval` unchanged, e.g. `--limit 50`. For XSum, `-T prompt_variant=short|long|teacher` is required and must match the prompt the model was trained on.

```bash
sbatch evaluate.sbatch xsum google/gemma-3-4b-it -T prompt_variant=short
sbatch evaluate.sbatch xsum ../models/gemma3-xsum-self-dist-short -T prompt_variant=short
sbatch evaluate.sbatch crime google/gemma-3-4b-it --adapter ../models/gemma3-crime-full-lora
```

The script always uses greedy decoding and loads the weights in bfloat16, so every model is compared at the same precision whatever its `config.json` says.

The `semantic` score needs `sentence-transformers/all-MiniLM-L6-v2` in the Hugging Face cache. Compute nodes have no outbound network, so download it once from a login node. If it is missing, `semantic` is reported as 0.0 with only a printed warning.

### Isambard Cuda technical details

Isambard-AI's GPU driver (565.57.01) only supports up to CUDA 12.7, but the torch that `uv sync` installs here is built for CUDA 13.0. `evaluate.sbatch` puts the CUDA 13.0 [forward-compatibility driver](https://docs.isambard.ac.uk/user-documentation/guides/gpus_and_cuda/#cuda-forward-compatibility) ahead of the system one. To run anything by hand, you need to have set it in your shell first:

```bash
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Without it nothing errors: `torch.cuda.is_available()` is `False` and the model quietly runs on the CPU, which is much slower.

## Running on Azure

`Azure/evaluate.sh` runs the same command on the Azure ML compute instance:

```bash
Azure/evaluate.sh <crime|xsum> <model> [--adapter <ref>] [inspect eval args...]
```

`<model>` and `--adapter` additionally accept an Azure ML registered model reference, `azureml:<name>:<version>`. The script downloads it into `evaluate/.azureml_models/` (skipped if already cached there) and records the reference in the log's metadata. The registry defaults to resource group `tire-1` and workspace `tire-2`; override them with `AZUREML_RG`, `AZUREML_WS` and `AZUREML_SUBSCRIPTION`.

```bash
Azure/evaluate.sh crime google/gemma-3-4b-it --adapter azureml:gemma3-crime-full-lora:1
```

## Running `inspect eval` directly

The scripts above only fill in arguments, so the tasks also run directly:

```bash
uv run inspect eval crime_task.py --model hf-peft/google/gemma-3-4b-it \
  -M adapter_path=../models/gemma3-crime-full-lora \
  -M batch_size=8 -M do_sample=false -M dtype=bfloat16
```

Pass `-M do_sample=false` and `-M dtype=bfloat16` every time. Inspect's HF provider samples by default, and a checkpoint whose config says float32 would load in fp32.

The `hf-peft` provider (`hf_peft_provider.py`) is registered with Inspect as a plugin through the entry point in `pyproject.toml`. `uv sync` installs it.
