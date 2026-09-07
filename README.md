# Model merging

## Overview

TBA

## Setup of code

This is somewhat specific to getting setup on Azure at the Turing.

### For EVERY new compute instance on Azure at Turing

1. Github setup

    ```bash
    git config --global user.name <your name>
    git config --global user.email <your email>
    ```

2. Ensure that you have your `HF_TOKEN` exported to Azure:

    ```bash
    echo 'export HF_TOKEN=<add HF token here>' >> ~/.bashrc
    source ~/.bashrc
    ```

    this ensures that you can download from Hugging Face without being rate-limited (and can use gated models).

3. Install [uv](https://docs.astral.sh/uv/) (used to manage dependencies for each sub-project below):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

4. Trained/merged models are pulled from the Azure ML model registry rather than committed to git (see [Model storage](#model-storage) below) — you'll need the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) with the `ml` extension, logged in and scoped to the `tire-2` workspace:

    ```bash
    az extension remove -n azure-cli-ml  # our Azure currently comes with v1 of azure-cli-ml and this conflicts with the new required ml extension
    az extension add -n ml
    az login
    az configure --defaults group=tire-1 workspace=tire-2
    ```

    Only the `ml` (v2) extension is needed — do not also install the legacy `azure-cli-ml` extension, the two conflict with each other.

### Setup the code if you have not pulled it from github before

1. Clone the repo:

    ```bash
    git clone https://github.com/alan-turing-institute/model-merging.git
    cd model-merging
    ```

2. If you are on the Turing Azure, then you need to do the symlink trick to make uv fast (as per the TIRE Azure user guide).

    ```bash
    mkdir ~/venvs && mkdir ~/venvs/model-merging
    mkdir ~/venvs/model-merging/train
    mkdir ~/venvs/model-merging/merge
    mkdir ~/venvs/model-merging/evaluate
    ```

    Now make the symlinks from the model-merging directory:

    ```bash
    ln –s ~/venvs/model-merging/train train/.venv
    ln –s ~/venvs/model-merging/merge merge/.venv
    ln –s ~/venvs/model-merging/evaluate evaluate/.venv
    ```

3. This repo isn't one Python project — `train/`, `merge/`, and `evaluate/` are each a separate uv-managed environment (with different, sometimes conflicting, dependency versions — e.g. `merge/` needs a patched `transformers` that `train/` doesn't). Sync each one you plan to use:

    ```bash
    cd train    && uv sync && cd ..
    cd merge    && uv sync && cd ..
    cd evaluate && uv sync && cd ..
    ```

    Commands in each sub-project's README assume you're running them via that sub-project's environment, e.g. `uv run axolotl train crime_gemma.yaml` from inside `train/`.

### Setup code if you have already pulled it once, but want to setup on a new compute instance

This should be used if you have previously pulled the code from github, but are starting a new compute instance and want to set it up.  We assume that you pulled it to `Users/ANOther/model-merging`.

1. To ensure that git works, go to the `model-merging` directory and run

    ```bash
    git status
    ```

    This will fail, claiming dubious ownership of the git repository (your other compute instance), but will suggest that you can add an exception by running a command, eg

    ```bash
    git config --global --add safe.directory <your model-merging directory>
    ```

    which you should run.

2. If you are on the Turing Azure, then you need to do the symlink trick to make uv fast (as per the TIRE Azure user guide).  Since you have previously pulled the code, you should have symlinks already, but these will be broken.  To check this, you can run

    ```bash
    find . -maxdepth 2 -xdev -xtype l
    ```

    You can then make the directories to where these symlinks already point, eg.

    ```bash
    mkdir ~/venvs && mkdir ~/venvs/model-merging
    mkdir ~/venvs/model-merging/train
    mkdir ~/venvs/model-merging/merge
    mkdir ~/venvs/model-merging/evaluate
    ```

    You do not need to remake the symlinks.

### Some things to note

1. A CUDA GPU is required for training/(merging)/evaluation (this project was developed against both an A100 80GB and a T4 setup) — there's no CPU-only path.  Merging can be done using CPUs, but is a bit quicker using GPUs.

2. To use gated models such as `google/gemma-3-4b-it`, you need to accept its license on the [model page](https://huggingface.co/google/gemma-3-4b-it) with your HF account, then authenticate locally so `transformers`/`axolotl` can download it:

    ```bash
    uv run --project evaluate huggingface-cli login
    ```

3. The root-level [prepare_data.py](prepare_data.py) and [Convert_to_full_model.py](Convert_to_full_model.py) scripts aren't tied to a project of their own — run them with the `evaluate` environment, which already has `datasets`/`transformers`/`peft`:

    ```bash
    uv run --project evaluate python Convert_to_full_model.py BASE_MODEL LORA_PATH OUTPUT_PATH
    ```

    [prepare_data.py](prepare_data.py) is more ad hoc, since different models expect inputs in different forms (conversational etc.) and also the question asked depends on the task and give dataset.  For this, run python in the environment and then copy/paste your own code into it:

    ```bash
    uv run --project evaluate python
    ```

## Model merging using mergekit

Overview of the process for a `poor-mans parallelism' model merging to train a classifier:

1. Split your dataset $D$ into parts $D_i$ - see [prepare data python script](prepare_data.py).
2. Train different copies of your base model on the $D_i$ to produce a LoRA adapter $L_i$ - see [train readme](train/README.md).
3. Use mergekit to combine the different $M_i$ into one model $\tilde{M}$ - see [merge readme](merge/README.md).
4. Test performance of $\tilde{M}$. - see [evaluate readme](evaluate/README.md).

## Merging a base model with its LoRA adapter

You should not need to combine a LoRA adapter with its base model in order to merge it, but if you do wish to do this, then you can using the [conversion script](Convert_to_full_model.py).
    This takes as input:

    - `BASE_MODEL` - a HG address, or path to the base model
    - `LORA_PATH` - the path of the LoRA adapter trained in step 3
    - `OUTPUT_PATH` - an output path for the combined model

    It should be run as

    ```bash
    uv run --project evaluate python Convert_to_full_model.py BASE_MODEL LORA_PATH OUTPUT_PATH
    ```

    (The evaluate environment already has some of the required dependencies.)

## Model storage

Trained/converted/merged models are large (GBs each) and are **not** kept in `models/` locally or committed to git (`models/` is gitignored) — they're registered as Model assets in the `tire-2` Azure ML workspace (resource group `tire-1`), backed by its default blob datastore. Register a new artifact with:

```bash
az ml model create --name <name> --version 1 --type custom_model --path <local-path> --resource-group tire-1 --workspace-name tire-2
```

and fetch one back (or pass `azureml:<name>:<version>` directly to [evaluate.py](evaluate/README.md), which downloads automatically) with:

```bash
az ml model download --name <name> --version 1 --download-path <dir> --resource-group tire-1 --workspace-name tire-2
```

To view the models currently stored on Azure:

```bash
az ml model list --resource-group tire-1 --workspace-name tire-2 --output table
```

Current registry name for each pipeline artifact:

| Pipeline artifact | Registered model name |
|---|---|
| LoRA adapter trained on the full dataset (`crime_gemma.yaml`) | `gemma3-crime-full-lora` |
| LoRA adapter trained on half 1 (`crime_gemma1.yaml`) | `gemma3-crime-1-of-2-lora` |
| LoRA adapter trained on half 2 (`crime_gemma2.yaml`) | `gemma3-crime-2-of-2-lora` |
| Full model = full-dataset adapter merged into the base model | `gemma3-crime-full` |
| Full model = half-1 adapter merged into the base model | `gemma3-crime-1-of-2` |
| Full model = half-2 adapter merged into the base model | `gemma3-crime-2-of-2` |
| Linear merge of the two half-dataset full models | `gemma3-crime-merged-linear-2` |
