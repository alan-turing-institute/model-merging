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
