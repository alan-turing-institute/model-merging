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

## Summarisation

`evaluate_summarisation.py` is the same script for a generation task: same
`--model` / `--adapter` / `azureml:` handling, same one-JSON-per-variant output
contract, but scored with ROUGE against the XSum reference summaries.

```bash
uv run python evaluate_summarisation.py \
  --model google/gemma-3-4b-it \
  --adapter azureml:gemma3-xsum-full-lora:1 \
  --output results/gemma3-xsum-full-lora.json
```

It reads the fully rendered prompt from the dataset's `prompt` column (written
by `prepare_xsum_data.py`) rather than rebuilding the prompt string, so
training and evaluation cannot drift apart.

Two things it reports beyond ROUGE:

- **Output length and sentence count.** A merge that has drifted back towards
  base behaviour gets chattier — preamble returns, one sentence becomes three —
  and that usually moves before ROUGE does, and is far easier to interpret.
- **A few generations, printed at the end of the run** (`--show N`, default 3).

`--batch-size` controls prompts per `generate()` call; lower it on CUDA OOM.
Generation is batched with left padding, so batch size does not change the
output. Compare result files with `experiments/bootstrap_rouge.py` — the paired
bootstrap counterpart to `experiments/mcnemar.py`. See [XSUM.md](../XSUM.md).
