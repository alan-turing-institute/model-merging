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

## Summarisation via Inspect AI (default)

`inspect-ai` was already a declared dependency here but nothing imported it.
The summarisation evaluation now runs through it.

| File | |
|---|---|
| `xsum_task.py` | the XSum task: dataset, `generate()` solver, ROUGE scorer |
| `hf_peft_provider.py` | an `hf-peft` model provider — the built-in `hf` one has no adapter support |
| `run_inspect_xsum.py` | wrapper: resolves `azureml:` refs, writes this repo's results JSON |

The pipelines use it by default (`XSUM_EVAL=legacy` selects the old
`evaluate_summarisation.py`). Both write the **same** results JSON, so
`experiments/bootstrap_rouge.py` and `experiments/crosstask_table.py` read
either without caring which produced it.

```bash
uv run python run_inspect_xsum.py \
  --model google/gemma-3-4b-it \
  --adapter azureml:gemma3-xsum-full-lora:3 \
  --output results/gemma3-xsum-full-lora.json
```

Run it from this directory — like `evaluate.py`, it depends on cwd for
`.azureml_models` and for locating the task file.

What using Inspect actually buys:

- **A browsable log per run.** `inspect view --log-dir <dir>` shows every
  sample, its prompt, completion and score. The pipelines write these next to
  the results JSON, and the JSON records the log path, so any number in a
  comparison table can be traced back to the samples behind it.
- **Standard error on ROUGE**, reported alongside the mean. The hand-rolled
  script never gave any measure of uncertainty, which mattered here: the crime
  arm's headline result rested on a gap smaller than its own training noise.
- **The task is reusable outside this repo's scripts**, via the normal CLI:

  ```bash
  inspect eval xsum_task.py --model hf/google/gemma-3-4b-it \
    -M batch_size=8 -M do_sample=false --max-tokens 64 -T limit=100
  ```

  Adapters are the exception. Inspect resolves `--model` *before* it imports
  the task file, so the `hf-peft` provider registered inside `xsum_task.py` is
  registered too late and `hf-peft/...` comes back as an unrecognised API. Use
  `run_inspect_xsum.py` for anything with an adapter — it imports the provider
  first.

Two details that keep results comparable with the legacy script: prompts come
from the dataset's `prompt` column (rendered once at data-prep time, so neither
evaluator re-derives the wording), and `do_sample=False` is set explicitly —
Inspect's HF provider defaults it to **true**, which would have quietly made
every comparison noisier than the one it was being compared against.
