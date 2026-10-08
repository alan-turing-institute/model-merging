# Readme for using Axolotl to train models

Here we give details and code of how to train a model to be a classifier on crime data.

## Data preparation

First we use the `prepare_data.py` script to split the crime dataset into three parts, for training, validation and testing, saved to `../datasets/crime_dataset`.

```bash
uv run python prepare_data.py
```

To train several models on disjoint parts of the data (e.g. for merging), split it into `n` pieces with [split_dataset.py](../split_dataset.py). This splits the `train` and `validation` parts and drops `test`, keeping it held out for evaluating the merged model:

```bash
uv run python ../split_dataset.py n ../datasets/crime_dataset ../datasets
```

which saves `../datasets/crime_dataset_<i>_of_<n>`.

To run with other datasets etc, edit the python script.

## Axolotl for training models

The parameters for training using Axolotl are controlled via a [yaml file](crime_gemma.yaml).  For full details see the [online documentation](https://docs.axolotl.ai/docs/getting-started.html).

The key parameters are:

- `base_model` - this can be a HF address, or a path to a model on disk
- `datasets` - again can be a HF address, or a path to a dataset on disk.  You can also specify different training and validation datasets here.

We provide an sbatch script and yaml config file to run training on the crime data.

```bash
sbatch train_crime.sbatch <config.yaml> <dataset_dir> <output_dir>
#   e.g. sbatch train_crime.sbatch crime_gemma.yaml \
#            ../datasets/crime_dataset_1_of_2 ../models/gemma3-4b-crime-1-of-2-lora
```

which trains the model detailed in `<config.yaml>` (and the parameters given their), on the dataset `<dataset_dit>` and writes the output to `<output_dir>`.

Alternatively, to run the training directly, write a properly configured yaml and run

```bash
uv run axolotl train crime_gemma.yaml
```

On Azure, the LoRA adapter should then be saved in the Azure blob storage

```bash
az ml model create --name <name> --version 1 --type custom_model --path <local-path> --resource-group tire-1 --workspace-name tire-2
```
