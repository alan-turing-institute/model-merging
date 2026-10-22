# Readme for evaluating trained/merged models

Evaluation runs as [Inspect AI](https://inspect.aisi.org.uk/) tasks:

- `crime_task.py` - classifies Reddit posts as `crime` / `not_crime`. Reports accuracy, precision, recall, F1, confusion-matrix counts and the share of unparseable answers.
- `wine_task.py` - predicts a wine's Wine Enthusiast points (80–100) from its review. Reports mean absolute error (with RMSE and the MAE of always answering 88, the bar to beat), Pearson/Spearman correlation with the true points, the spread of predictions, bias and MAE per score band, accuracy within 0/1/2 points and the share of answers off the 80–100 scale. Answers with no number are left out of all of these and reported separately as `unanswered`.
- `xsum_task.py` - one-sentence XSum summaries. Reports ROUGE-1/2/L, embedding similarity to the reference (`semantic`), whether the names and numbers in the summary appear in the article (`entity_support`), and output length.

All three build their prompts with the same function as the script that made the training data (`train/prepare_data_crime.py`, `train/prepare_data_wine.py` and `self-distill/prepare_data_XSum.py`), so training and evaluation wording cannot drift apart.

Each run writes a `.eval` log to `logs/` with every sample, score and setting. Browse them with:

```bash
uv run inspect view --log-dir logs
```

and load them into pandas with `inspect_ai.analysis.samples_df("logs")` / `evals_df("logs")`.

`evaluate.sbatch` refuses a model whose `provenance.json` says its training went NaN (`health.ok` false), since it can only score NaN; set `ALLOW_BROKEN_MODEL=1` to evaluate it anyway. It also stores what the model is in each log's metadata, from the model's (or adapter's) `provenance.json`: `model_kind` (`trained`, `merged` or `base`), method, base model, training data and its hash, teacher and targets, `model_coverage`, hyperparameters, final losses, or for a merge its method, inputs and weights (see [`../provenance.py`](../provenance.py)). `evals_df` gives the metadata as a JSON string, so expand it into columns with

```python
import json, pandas as pd
from inspect_ai.analysis import evals_df

df = evals_df("logs")
meta = pd.json_normalize(df.metadata.map(json.loads))  # model_kind, targets, model_coverage, ...
```

Logs written before this was added were given the same metadata afterwards with Inspect's `edit_eval_log`, so it is in each log's top-level `metadata` (what `evals_df` reads) rather than `eval.metadata`, and the edit is recorded in its `log_updates`.

## Running on Isambard-AI

Submit `evaluate.sbatch` from `evaluate/`:

```bash
sbatch evaluate.sbatch <crime|wine|xsum> <model> [--adapter <dir>] [inspect eval args...]
```

`<model>` is a Hugging Face hub id or a full model saved on disk. `--adapter` puts a LoRA adapter on top of the model; this is only needed for the crime and wine models, as the XSum models are self-distilled full models. Anything after that goes to `inspect eval` unchanged, e.g. `--limit 50`. For XSum, `-T prompt_variant=short|long|teacher` is required and must match the prompt the model was trained on.

```bash
sbatch evaluate.sbatch xsum google/gemma-3-4b-it -T prompt_variant=short
sbatch evaluate.sbatch xsum ../models/gemma3-xsum-self-dist-short -T prompt_variant=short
sbatch evaluate.sbatch crime google/gemma-3-4b-it --adapter ../models/gemma3-crime-full-lora
sbatch evaluate.sbatch wine google/gemma-3-4b-it --adapter ../models/gemma3-4b-wine-full-lora
```

The script always uses greedy decoding and loads the weights in bfloat16, so every model is compared at the same precision whatever its `config.json` says.

The `semantic` score needs `sentence-transformers/all-MiniLM-L6-v2` in the Hugging Face cache. Compute nodes have no outbound network, so download it once from a login node. If it is missing, `semantic` is reported as 0.0 with only a printed warning.


## Running on Azure

`Azure/evaluate.sh` runs the same command on the Azure ML compute instance:

```bash
Azure/evaluate.sh <crime|wine|xsum|translation> <model> [--adapter <ref>] [inspect eval args...]
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
