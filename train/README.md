# Readme for using Axolotl to train models

Here we give details and code of how to train a model to be a classifier on crime data.

## Data preparation

First we use the `prepare_data_crime.py` script to split the crime dataset into three parts, for training, validation and testing, saved to `../datasets/crime_dataset`.

```bash
uv run python prepare_data_crime.py
```

To train several models on disjoint parts of the data (e.g. for merging), split it into `n` pieces with [split_dataset.py](../split_dataset.py). This splits the `train` and `validation` parts and drops `test`, keeping it held out for evaluating the merged model:

```bash
uv run python ../split_dataset.py n ../datasets/crime_dataset datasets
```

which saves `datasets/crime_dataset_<i>_of_<n>`. The prepare scripts write a `provenance.json` with `coverage: "full"`, and each piece gets one with e.g. `coverage: "1 of 2"`, the source dataset and its record, the seed and the source's row counts (see [`../provenance.py`](../provenance.py)).

[prepare_data_wine.py](prepare_data_wine.py) does the same for wine reviews (predicting the 80–100 points rating from the review text), sampling 10k/1k/1k rows to `../datasets/wine_dataset` after removing duplicate reviews (`--help` for options).

To run with other datasets etc, edit the python script.

## Axolotl for training models

The parameters for training using Axolotl are controlled via a [yaml file](crime_gemma.yaml).  For full details see the [online documentation](https://docs.axolotl.ai/docs/getting-started.html).

The key parameters are:

- `base_model` - this can be a HF address, or a path to a model on disk
- `datasets` - again can be a HF address, or a path to a dataset on disk.  You can also specify different training and validation datasets here.

We provide an sbatch script, and yaml config files for the crime ([crime_gemma.yaml](crime_gemma.yaml)) and wine ([wine_gemma.yaml](wine_gemma.yaml)) tasks, to run training.

```bash
sbatch train.sbatch <config.yaml> <dataset_dir> <output_dir>
#   e.g. sbatch train.sbatch crime_gemma.yaml \
#            ../datasets/crime_dataset_1_of_2 ../models/gemma3-4b-crime-1-of-2-lora
#        sbatch train.sbatch wine_gemma.yaml \
#            ../datasets/wine_dataset ../models/gemma3-4b-wine-full
```

which trains the model detailed in `<config.yaml>` (and the parameters given there), on the dataset `<dataset_dir>` and writes the output to `<output_dir>`. The log is saved in `<output_dir>` as `<config name>-<jobid>.out`, e.g. `wine_gemma-1234567.out`. Once training succeeds, `<output_dir>/provenance.json` records the base model, method, key hyperparameters, final train/eval loss and the training data with its own record, including its `coverage` (see [`../provenance.py`](../provenance.py)).

Alternatively, to run the training directly, write a properly configured yaml and run

```bash
uv run axolotl train crime_gemma.yaml
```

On Azure, the LoRA adapter should then be saved in the Azure blob storage

```bash
az ml model create --name <name> --version 1 --type custom_model --path <local-path> --resource-group tire-1 --workspace-name tire-2
```
