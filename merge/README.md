# Readme for using mergekit for model merging

The models listed in these configs are read from local paths — mergekit has no notion of the Azure ML model registry, so inputs need to be downloaded first. Models for this project are registered in the `tire-2` Azure ML workspace (see the root [README](../README.md) for the name mapping), not kept in `models/` locally.

## `linear` — weighted average of full model weights

```bash
az ml model download --name gemma3-crime-1-of-2 --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
az ml model download --name gemma3-crime-2-of-2 --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2

uv run mergekit-yaml merge_linear_config.yaml ../models/gemma3-crime-merged-linear-2 --cuda --lazy-unpickle --allow-crimes
```

## Other merge methods

[merge_task_arithmetic_config.yaml](merge_task_arithmetic_config.yaml), [merge_ties_config.yaml](merge_ties_config.yaml), [merge_dare_ties_config.yaml](merge_dare_ties_config.yaml), [merge_model_stock_config.yaml](merge_model_stock_config.yaml), [merge_slerp_config.yaml](merge_slerp_config.yaml), and [merge_arcee_fusion_config.yaml](merge_arcee_fusion_config.yaml) merge the two **LoRA adapters** directly, using mergekit's `base_model+adapter_path` reference syntax, rather than the pre-materialized full models — see the comment at the top of each file for what the method does and why it's worth trying. Download the adapters instead:

```bash
az ml model download --name gemma3-crime-1-of-2-lora --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
az ml model download --name gemma3-crime-2-of-2-lora --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
```

Because these configs reference LoRA adapters, mergekit needs somewhere to materialize each `base_model+adapter` combination before merging — pass `--lora-merge-cache` (skipped/reused if already cached there):

```bash
uv run mergekit-yaml merge_task_arithmetic_config.yaml ../models/gemma3-crime_merged-task-arithmetic --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache
uv run mergekit-yaml merge_ties_config.yaml            ../models/gemma3-crime-merged-ties            --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache
uv run mergekit-yaml merge_dare_ties_config.yaml       ../models/gemma3-crime-merged-dare-ties        --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache
uv run mergekit-yaml merge_model_stock_config.yaml     ../models/gemma3-crime-merged-model-stock      --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache
uv run mergekit-yaml merge_slerp_config.yaml           ../models/gemma3-crime-merged-slerp            --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache
uv run mergekit-yaml merge_arcee_fusion_config.yaml    ../models/gemma3-crime-merged-arcee-fusion     --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache
```

`task_arithmetic`/`ties`/`dare_ties`/`model_stock` also need `base_model: google/gemma-3-4b-it` in the config (to compute each adapter's weight delta) — this downloads from the Hugging Face hub automatically the first time, same as training.

## After merging

Register the result and clean up the local copies:

```bash
az ml model create --name <new-model-name> --version 1 --type custom_model --path ../models/<merged-dir> --resource-group tire-1 --workspace-name tire-2
rm -rf ../models
```
