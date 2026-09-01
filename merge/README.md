# Readme for using mergekit for model merging

The models listed in [merge_linear_config.yaml](merge_linear_config.yaml) are read from local paths — mergekit has no notion of the Azure ML model registry, so the inputs need to be downloaded first. Models for this project are registered in the `tire-2` Azure ML workspace (see the root [README](../README.md) for the name mapping), not kept in `models/` locally.

Download the two inputs the current config expects

```bash
az ml model download --name gemma3-crime-1-of-2 --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
az ml model download --name gemma3-crime-2-of-2 --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
```

Merge

```bash
uv run mergekit-yaml merge_linear_config.yaml ../models/merged_linear --cuda --lazy-unpickle --allow-crimes
```

Once you're happy with the result, register it and remove the local copies:

```bash
az ml model create --name <new-model-name> --version 1 --type custom_model --path ../models/merged_linear --resource-group tire-1 --workspace-name tire-2
rm -rf ../models
```
