# Readme for using Axolotl to train models

The parameters for training using Axolotl are controlled via a [yaml file](crime_gemma.yaml).  For full details see the [online documentation](https://docs.axolotl.ai/docs/getting-started.html).

The key parameters are:

- `base_model` - this can be a HF address, or a path to a model on disk
- `datasets` - again can be a HF address, or a path to a dataset on disk.  You can also specify different training and validation datasets here.

Once you have a properly configured yaml, to train the model, simply run

```bash
uv run axolotl train crime_gemma.yaml
```

The LoRA adapter should then be saved in the Azure blob storage

```bash
az ml model create --name <name> --version 1 --type custom_model --path <local-path> --resource-group tire-1 --workspace-name tire-2
```

## XSum summarisation configs

`xsum_gemma.yaml` and `xsum_gemma{1,2}.yaml` are the summarisation arm's
equivalents of `crime_gemma*.yaml` — see [XSUM.md](../XSUM.md). They add
`gradient_checkpointing: true` (without which these settings OOM on a T4's
16GB), state `train_on_inputs: false` explicitly, and use 2 epochs rather
than 3.

Run `python prepare_xsum_data.py` from the repo root first — the configs read
`../../datasets/xsum_dataset*`.
