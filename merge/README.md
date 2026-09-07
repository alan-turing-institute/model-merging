# Readme for using mergekit for model merging

The models listed in these configs are read from local paths — mergekit has no notion of the Azure ML model registry, so inputs need to be downloaded first.

## Download the models to be merged from Azure

```bash
az ml model download --name gemma3-crime-1-of-2-lora --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
az ml model download --name gemma3-crime-2-of-2-lora --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
```

## Running a merge using mergekit

Merges in mergekit are controlled by a yaml file.  For example, the current yaml file describes a linear merge between two LoRA adapters for the base model `gemma3-4b-it` weighted 50/50:

```yaml
{
# config.yaml — a minimal linear merge of two LoRA adapters on top of the same base model.
merge_method: linear        # linear merge of two models, weighted by the "weight" parameter below

models:                     # list of models to merge
  - model: "google/gemma-3-4b-it+../models/gemma3-crime-1-of-2-lora"
    parameters:
      weight: 0.5

  - model: "google/gemma-3-4b-it+../models/gemma3-crime-2-of-2-lora"
    parameters:
      weight: 0.5

# parameters: insert parameters here
tokenizer_source: union     # union of tokenizers from all models

dtype: float16
}
```

You should edit this file to describe the merge you want to complete.

- `merge_method`: There are several different merge methods to chose from, e.g. linear, SLERP, Task Arithmetic, TIES, DARE, Arcee Fusion and more.  For a full list see the [merge methods section for mergekit](https://github.com/arcee-ai/mergekit/tree/main#merge-methods).  Note that the particular method used may require parameters to be set either for each model given, or a general parameter section outside the `models` section.  For example, `linear` merge requires a `weight` parameter for each model.
- `models`: List the models to be merged along with any required parameters for the merge method.  The models can be listed either by their Hugging Face ID, or by a path to their location on disk.  *Most usefully* (and not in the manual?), you can use specify the base model and LoRA adapter `<base_model>+<LoRA_adapater>`.  Note that both of these can be given as either Hugging Face IDs, or paths, e.g. above we use a mix of a HF ID for the base model and a path for the LoRA adapter.

`Mergekit` can also do more complicated multi-stage merging workflows.  These aren't covered by these scripts, but details can be found in the [multi-stage section in mergekit](https://github.com/arcee-ai/mergekit/tree/main#multi-stage-merging-mergekit-multi).

We give some examples of mergekit yaml files for different merging methods including [linear](merge_linear_config.yaml), [merge_task_arithmetic_config.yaml](merge_task_arithmetic_config.yaml), [merge_ties_config.yaml](merge_ties_config.yaml), [merge_dare_ties_config.yaml](merge_dare_ties_config.yaml), [merge_model_stock_config.yaml](merge_model_stock_config.yaml), [merge_slerp_config.yaml](merge_slerp_config.yaml), and [merge_arcee_fusion_config.yaml](merge_arcee_fusion_config.yaml).

To merge two complete models (not base model + LoRA adapter), then run

```bash
uv run mergekit-yaml <config yaml> <output directory>   --cuda --lazy-unpickle --allow-crimes
```

If your models are listed using the `base_model+adapter` syntax, then mergekit needs a place to materialize each `base_model+adapter` combination before merging.  To do this you must also pass a `--lora-merge-cache` path.

```bash
uv run mergekit-yaml merge_linear_config.yaml ../models/gemma3-crime_merged-linear --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache
```

## Possible problems

If the compute has GPUs with a smaller amount of VRAM, eg 16Gb, then this might max out, e.g. with TIES and DARE-TIES.  In this case, remove the `cuda` flag and instead use the system RAM.  Using CPUs, you can add the `-j` flag, to parallelise across several nodes.

```bash
uv run mergekit-yaml merge_ties_config.yaml  ../models/gemma3-crime-merged-ties --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache -j 8
```

`task_arithmetic`/`ties`/`dare_ties`/`model_stock` also need `base_model: google/gemma-3-4b-it` in the config (to compute each adapter's weight delta) — this downloads from the Hugging Face hub automatically the first time, same as training.

## After merging

Register the result and clean up the local copies:

```bash
az ml model create --name <new-model-name> --version 1 --type custom_model --path ../models/<merged-dir> --resource-group tire-1 --workspace-name tire-2
rm -rf ../models
```

## XSum summarisation configs

`merge_*_xsum_config.yaml` are the same merges for the XSum arm
(see [XSUM.md](../XSUM.md)). They differ from the crime configs in one way that
matters operationally: they merge the two half-data **adapters** via the
`<base_model>+<adapter>` syntax, so the halves never have to be converted into
full models first. That saves two conversions and ~17GB of scratch, but it does
mean `--lora-merge-cache` is required rather than optional:

```bash
uv run mergekit-yaml merge_linear_xsum_config.yaml ../models/gemma3-xsum-merged-linear \
  --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/.lora_merge_cache
```

`merge_task_arithmetic_w1_xsum_config.yaml` is not a method variant so much as a
diagnostic: it sets both weights to 1.0 instead of 0.5, which keeps both deltas
at full size. It exists to test whether a disappointing 0.5/0.5 merge is
under-adapted (halved deltas pulling the model back towards base) rather than
genuinely conflicted.
