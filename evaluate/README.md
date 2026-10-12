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

## Test data

`xsum_task.py` needs no separate preparation. If `datasets/xsum_prompts/test` is missing, it rebuilds the same 1,000 test articles `self-distill/prepare_data_XSum.py` selects, from the Hub at a pinned revision, and saves them there. Compute nodes have no network, so on Isambard build it once on a login node first:

```bash
uv run python xsum_task.py
```

**Reproducing the report's XSum numbers.** Those were produced from `prepare_xsum_data.py` on the `xsum-summarisation` branch, which collapsed whitespace and cut each article to 512 words; `prepare_data_XSum.py` on `main` does neither. Add `-T document_words=512` to score exactly the inputs the report used (checked identical for all 1,000 rows). Like `prompt_variant`, it must match what the model was trained on.

The optional reference-free metrics (`-T faithfulness=true`, see `xsum_faithfulness.py`) load about 6 GB of extra models and are off by default.

`crime_task.py` reads the split written by `train/prepare_data.py`.

## Checking a setup

```bash
bash smoke_test.sh
```

runs every task end to end on three synthetic rows with Inspect's built-in `mockllm` provider - no GPU, no gated model, no prepared data - and checks each writes a successful log. It is the quickest way to confirm a fresh clone works, and CI runs it on every pull request, together with `uv lock --check` for each environment.

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
