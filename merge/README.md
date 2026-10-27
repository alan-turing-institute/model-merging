# Readme for using mergekit for model merging

The models listed in these configs are read from local paths — mergekit has no notion of the Azure ML model registry, so inputs need to be downloaded first.

## Download the models to be merged from Azure

```bash
az ml model download --name gemma3-crime-1-of-2-lora --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
az ml model download --name gemma3-crime-2-of-2-lora --version 1 --download-path ../models --resource-group tire-1 --workspace-name tire-2
```

These are the original 4-bit adapters. The crime configs here expect the bf16 halves trained with [`../train/train.sbatch`](../train/README.md) at `../models/gemma3_4b_crime_{1,2}_of_2`; a downloaded adapter has to be moved there to be used instead.

## Running a merge using mergekit

Merges in mergekit are controlled by a yaml file.  For example, the current yaml file describes a linear merge between two LoRA adapters for the base model `gemma3-4b-it` weighted 50/50:

```yaml
{
# config.yaml — a minimal linear merge of two LoRA adapters on top of the same base model.
merge_method: linear        # linear merge of two models, weighted by the "weight" parameter below

models:                     # list of models to merge
  - model: "google/gemma-3-4b-it+../models/crime/split_2/gemma3_4b_crime_1_of_2"
    parameters:
      weight: 0.5

  - model: "google/gemma-3-4b-it+../models/crime/split_2/gemma3_4b_crime_2_of_2"
    parameters:
      weight: 0.5

# parameters: insert parameters here
tokenizer_source: union     # union of tokenizers from all models

dtype: bfloat16
}
```

You should edit this file to describe the merge you want to complete.

- `merge_method`: There are several different merge methods to chose from, e.g. linear, SLERP, Task Arithmetic, TIES, DARE, Arcee Fusion and more.  For a full list see the [merge methods section for mergekit](https://github.com/arcee-ai/mergekit/tree/main#merge-methods).  Note that the particular method used may require parameters to be set either for each model given, or a general parameter section outside the `models` section.  For example, `linear` merge requires a `weight` parameter for each model.
- `models`: List the models to be merged along with any required parameters for the merge method.  The models can be listed either by their Hugging Face ID, or by a path to their location on disk.  *Most usefully* (and not in the manual?), you can use specify the base model and LoRA adapter `<base_model>+<LoRA_adapater>`.  Note that both of these can be given as either Hugging Face IDs, or paths, e.g. above we use a mix of a HF ID for the base model and a path for the LoRA adapter.

`Mergekit` can also do more complicated multi-stage merging workflows.  These aren't covered by these scripts, but details can be found in the [multi-stage section in mergekit](https://github.com/arcee-ai/mergekit/tree/main#multi-stage-merging-mergekit-multi).

We give some examples of mergekit yaml files for different merging methods including [linear](crime/crime_merge_2_linear_config.yaml), [crime/crime_merge_2_task_arithmetic_config.yaml](crime/crime_merge_2_task_arithmetic_config.yaml), [crime/crime_merge_2_ties_config.yaml](crime/crime_merge_2_ties_config.yaml), [crime/crime_merge_2_dare_ties_config.yaml](crime/crime_merge_2_dare_ties_config.yaml), [crime/crime_merge_2_model_stock_config.yaml](crime/crime_merge_2_model_stock_config.yaml), [crime/crime_merge_2_slerp_config.yaml](crime/crime_merge_2_slerp_config.yaml), and [crime/crime_merge_2_arcee_fusion_config.yaml](crime/crime_merge_2_arcee_fusion_config.yaml).

To merge two complete models (not base model + LoRA adapter), then run

```bash
uv run mergekit-yaml <config yaml> <output directory>   --cuda --lazy-unpickle --allow-crimes
```

If your models are listed using the `base_model+adapter` syntax, then mergekit needs a place to materialize each `base_model+adapter` combination before merging.  To do this you must also pass a `--lora-merge-cache` path.

```bash
uv run mergekit-yaml crime/crime_merge_2_linear_config.yaml ../models/crime/split_2/gemma3_4b_crime_merge_2_linear --cuda --lazy-unpickle --allow-crimes --lora-merge-cache ../models/crime/split_2/.lora_merge_cache
```

### As a Slurm job, with a provenance record

[merge.sbatch](merge.sbatch) runs the same `mergekit-yaml` command as a job, from `merge/`, passing on any extra arguments (e.g. `--lora-merge-cache`):

```bash
sbatch merge.sbatch xsum/xsum_merge_2_linear_config.yaml ../models/xsum/split_2/gemma3_4b_xsum_short_merge_2_linear
```

The log goes to `<output_dir>/<config name>-<jobid>.out`, and once the merge succeeds `<output_dir>/provenance.json` records the merge method and each input model with its weight and its own record (training data, coverage, losses; see [`../provenance.py`](../provenance.py)). Its `coverage` lists the inputs', e.g. `["1 of 2", "2 of 2"]`. Before that, [`../fix_eos.py`](../fix_eos.py) gives the merge the stop tokens of its base (or first) model: mergekit writes no `generation_config.json`, and a `config.json` copied from a trained model has lost Gemma's `<end_of_turn>`, so without it the merge never stops generating.

## Possible problems

If the compute has GPUs with a smaller amount of VRAM, eg 16Gb, then this might max out, e.g. with TIES and DARE-TIES.  In this case, remove the `cuda` flag and instead use the system RAM.  Using CPUs, you can add the `-j` flag, to parallelise across several nodes.

```bash
uv run mergekit-yaml crime/crime_merge_2_ties_config.yaml  ../models/crime/split_2/gemma3_4b_crime_merge_2_ties --lazy-unpickle --allow-crimes --lora-merge-cache ../models/crime/split_2/.lora_merge_cache -j 8
```

`task_arithmetic`/`ties`/`dare_ties`/`model_stock` also need `base_model: google/gemma-3-4b-it` in the config (to compute each adapter's weight delta) — this downloads from the Hugging Face hub automatically the first time, same as training.

## After merging

Register the result and clean up the local copies:

```bash
az ml model create --name <new-model-name> --version 1 --type custom_model --path ../models/<merged-dir> --resource-group tire-1 --workspace-name tire-2
rm -rf ../models
```
