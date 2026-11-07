# Archives on Azure

Container `azureml-blobstore-064c9079-e9d7-4131-93b9-84f72e0c0e68`, folder `model-merging/tars/`. Made by
`make_and_upload.sbatch` from `plan.tsv` (`make_plan.sh`). Paths inside are relative to the `model-merging`
project directory; extract with `zstd -dc <name>.tar.zst | tar -C <model-merging dir> -xf -`.
Each archive has `<name>.files.txt` (every file, with sizes), `<name>.sha256` and `<name>.md5`.

Left out: the regenerable merge caches (`.lora_merge_cache*`, `intermediate/`), non-MLP Mistral and Llama
models, Gemma's attention models for wine and CLINC150, and the runs from before the seed repeats.

## datasets.tar.zst (~11 GB uncompressed)

- `datasets/ (all)`

## wine_mlp.tar.zst (25 models, ~159 GB uncompressed)

- `models/wine/seed_1/gemma3_4b_wine_mlp`
- `models/wine/seed_1/llama3_8b_wine_mlp`
- `models/wine/seed_1/llama3_8b_wine_mlp_full`
- `models/wine/seed_1/mistral_7b_wine_mlp`
- `models/wine/seed_1/mistral_7b_wine_mlp_full`
- `models/wine/seed_2/gemma3_4b_wine_mlp`
- `models/wine/seed_2/llama3_8b_wine_mlp`
- `models/wine/seed_2/llama3_8b_wine_mlp_full`
- `models/wine/seed_2/mistral_7b_wine_mlp`
- `models/wine/seed_2/mistral_7b_wine_mlp_full`
- `models/wine/seed_3/gemma3_4b_wine_mlp`
- `models/wine/seed_3/llama3_8b_wine_mlp`
- `models/wine/seed_3/llama3_8b_wine_mlp_full`
- `models/wine/seed_3/mistral_7b_wine_mlp`
- `models/wine/seed_3/mistral_7b_wine_mlp_full`
- `models/wine/seed_4/gemma3_4b_wine_mlp`
- `models/wine/seed_4/llama3_8b_wine_mlp`
- `models/wine/seed_4/llama3_8b_wine_mlp_full`
- `models/wine/seed_4/mistral_7b_wine_mlp`
- `models/wine/seed_4/mistral_7b_wine_mlp_full`
- `models/wine/seed_5/gemma3_4b_wine_mlp`
- `models/wine/seed_5/llama3_8b_wine_mlp`
- `models/wine/seed_5/llama3_8b_wine_mlp_full`
- `models/wine/seed_5/mistral_7b_wine_mlp`
- `models/wine/seed_5/mistral_7b_wine_mlp_full`

## clinc150_mlp.tar.zst (25 models, ~160 GB uncompressed)

- `models/clinc150/seed_1/gemma3_4b_clinc150_labels_mlp`
- `models/clinc150/seed_1/llama3_8b_clinc150_labels_mlp`
- `models/clinc150/seed_1/llama3_8b_clinc150_labels_mlp_full`
- `models/clinc150/seed_1/mistral_7b_clinc150_labels_mlp`
- `models/clinc150/seed_1/mistral_7b_clinc150_labels_mlp_full`
- `models/clinc150/seed_2/gemma3_4b_clinc150_labels_mlp`
- `models/clinc150/seed_2/llama3_8b_clinc150_labels_mlp`
- `models/clinc150/seed_2/llama3_8b_clinc150_labels_mlp_full`
- `models/clinc150/seed_2/mistral_7b_clinc150_labels_mlp`
- `models/clinc150/seed_2/mistral_7b_clinc150_labels_mlp_full`
- `models/clinc150/seed_3/gemma3_4b_clinc150_labels_mlp`
- `models/clinc150/seed_3/llama3_8b_clinc150_labels_mlp`
- `models/clinc150/seed_3/llama3_8b_clinc150_labels_mlp_full`
- `models/clinc150/seed_3/mistral_7b_clinc150_labels_mlp`
- `models/clinc150/seed_3/mistral_7b_clinc150_labels_mlp_full`
- `models/clinc150/seed_4/gemma3_4b_clinc150_labels_mlp`
- `models/clinc150/seed_4/llama3_8b_clinc150_labels_mlp`
- `models/clinc150/seed_4/llama3_8b_clinc150_labels_mlp_full`
- `models/clinc150/seed_4/mistral_7b_clinc150_labels_mlp`
- `models/clinc150/seed_4/mistral_7b_clinc150_labels_mlp_full`
- `models/clinc150/seed_5/gemma3_4b_clinc150_labels_mlp`
- `models/clinc150/seed_5/llama3_8b_clinc150_labels_mlp`
- `models/clinc150/seed_5/llama3_8b_clinc150_labels_mlp_full`
- `models/clinc150/seed_5/mistral_7b_clinc150_labels_mlp`
- `models/clinc150/seed_5/mistral_7b_clinc150_labels_mlp_full`

## crime_seed_1.tar.zst (36 models, ~201 GB uncompressed)

- `models/crime/seed_1/gemma3_4b_crime`
- `models/crime/seed_1/split_2/gemma3_4b_crime_1_of_2`
- `models/crime/seed_1/split_2/gemma3_4b_crime_2_of_2`
- `models/crime/seed_1/split_2/gemma3_4b_crime_merge_2_arcee_fusion`
- `models/crime/seed_1/split_2/gemma3_4b_crime_merge_2_dare_ties`
- `models/crime/seed_1/split_2/gemma3_4b_crime_merge_2_linear`
- `models/crime/seed_1/split_2/gemma3_4b_crime_merge_2_model_stock`
- `models/crime/seed_1/split_2/gemma3_4b_crime_merge_2_slerp`
- `models/crime/seed_1/split_2/gemma3_4b_crime_merge_2_task_arithmetic`
- `models/crime/seed_1/split_2/gemma3_4b_crime_merge_2_ties`
- `models/crime/seed_1/split_4/gemma3_4b_crime_1_of_4`
- `models/crime/seed_1/split_4/gemma3_4b_crime_2_of_4`
- `models/crime/seed_1/split_4/gemma3_4b_crime_3_of_4`
- `models/crime/seed_1/split_4/gemma3_4b_crime_4_of_4`
- `models/crime/seed_1/split_4/gemma3_4b_crime_merge_4_arcee_fusion`
- `models/crime/seed_1/split_4/gemma3_4b_crime_merge_4_dare_ties`
- `models/crime/seed_1/split_4/gemma3_4b_crime_merge_4_linear`
- `models/crime/seed_1/split_4/gemma3_4b_crime_merge_4_model_stock`
- `models/crime/seed_1/split_4/gemma3_4b_crime_merge_4_slerp`
- `models/crime/seed_1/split_4/gemma3_4b_crime_merge_4_task_arithmetic`
- `models/crime/seed_1/split_4/gemma3_4b_crime_merge_4_ties`
- `models/crime/seed_1/split_8/gemma3_4b_crime_1_of_8`
- `models/crime/seed_1/split_8/gemma3_4b_crime_2_of_8`
- `models/crime/seed_1/split_8/gemma3_4b_crime_3_of_8`
- `models/crime/seed_1/split_8/gemma3_4b_crime_4_of_8`
- `models/crime/seed_1/split_8/gemma3_4b_crime_5_of_8`
- `models/crime/seed_1/split_8/gemma3_4b_crime_6_of_8`
- `models/crime/seed_1/split_8/gemma3_4b_crime_7_of_8`
- `models/crime/seed_1/split_8/gemma3_4b_crime_8_of_8`
- `models/crime/seed_1/split_8/gemma3_4b_crime_merge_8_arcee_fusion`
- `models/crime/seed_1/split_8/gemma3_4b_crime_merge_8_dare_ties`
- `models/crime/seed_1/split_8/gemma3_4b_crime_merge_8_linear`
- `models/crime/seed_1/split_8/gemma3_4b_crime_merge_8_model_stock`
- `models/crime/seed_1/split_8/gemma3_4b_crime_merge_8_slerp`
- `models/crime/seed_1/split_8/gemma3_4b_crime_merge_8_task_arithmetic`
- `models/crime/seed_1/split_8/gemma3_4b_crime_merge_8_ties`

## crime_seed_1_mlp.tar.zst (5 models, ~32 GB uncompressed)

- `models/crime/seed_1/gemma3_4b_crime_mlp`
- `models/crime/seed_1/llama3_8b_crime_mlp`
- `models/crime/seed_1/llama3_8b_crime_mlp_full`
- `models/crime/seed_1/mistral_7b_crime_mlp`
- `models/crime/seed_1/mistral_7b_crime_mlp_full`

## crime_seed_2.tar.zst (36 models, ~201 GB uncompressed)

- `models/crime/seed_2/gemma3_4b_crime`
- `models/crime/seed_2/split_2/gemma3_4b_crime_1_of_2`
- `models/crime/seed_2/split_2/gemma3_4b_crime_2_of_2`
- `models/crime/seed_2/split_2/gemma3_4b_crime_merge_2_arcee_fusion`
- `models/crime/seed_2/split_2/gemma3_4b_crime_merge_2_dare_ties`
- `models/crime/seed_2/split_2/gemma3_4b_crime_merge_2_linear`
- `models/crime/seed_2/split_2/gemma3_4b_crime_merge_2_model_stock`
- `models/crime/seed_2/split_2/gemma3_4b_crime_merge_2_slerp`
- `models/crime/seed_2/split_2/gemma3_4b_crime_merge_2_task_arithmetic`
- `models/crime/seed_2/split_2/gemma3_4b_crime_merge_2_ties`
- `models/crime/seed_2/split_4/gemma3_4b_crime_1_of_4`
- `models/crime/seed_2/split_4/gemma3_4b_crime_2_of_4`
- `models/crime/seed_2/split_4/gemma3_4b_crime_3_of_4`
- `models/crime/seed_2/split_4/gemma3_4b_crime_4_of_4`
- `models/crime/seed_2/split_4/gemma3_4b_crime_merge_4_arcee_fusion`
- `models/crime/seed_2/split_4/gemma3_4b_crime_merge_4_dare_ties`
- `models/crime/seed_2/split_4/gemma3_4b_crime_merge_4_linear`
- `models/crime/seed_2/split_4/gemma3_4b_crime_merge_4_model_stock`
- `models/crime/seed_2/split_4/gemma3_4b_crime_merge_4_slerp`
- `models/crime/seed_2/split_4/gemma3_4b_crime_merge_4_task_arithmetic`
- `models/crime/seed_2/split_4/gemma3_4b_crime_merge_4_ties`
- `models/crime/seed_2/split_8/gemma3_4b_crime_1_of_8`
- `models/crime/seed_2/split_8/gemma3_4b_crime_2_of_8`
- `models/crime/seed_2/split_8/gemma3_4b_crime_3_of_8`
- `models/crime/seed_2/split_8/gemma3_4b_crime_4_of_8`
- `models/crime/seed_2/split_8/gemma3_4b_crime_5_of_8`
- `models/crime/seed_2/split_8/gemma3_4b_crime_6_of_8`
- `models/crime/seed_2/split_8/gemma3_4b_crime_7_of_8`
- `models/crime/seed_2/split_8/gemma3_4b_crime_8_of_8`
- `models/crime/seed_2/split_8/gemma3_4b_crime_merge_8_arcee_fusion`
- `models/crime/seed_2/split_8/gemma3_4b_crime_merge_8_dare_ties`
- `models/crime/seed_2/split_8/gemma3_4b_crime_merge_8_linear`
- `models/crime/seed_2/split_8/gemma3_4b_crime_merge_8_model_stock`
- `models/crime/seed_2/split_8/gemma3_4b_crime_merge_8_slerp`
- `models/crime/seed_2/split_8/gemma3_4b_crime_merge_8_task_arithmetic`
- `models/crime/seed_2/split_8/gemma3_4b_crime_merge_8_ties`

## crime_seed_2_mlp.tar.zst (5 models, ~32 GB uncompressed)

- `models/crime/seed_2/gemma3_4b_crime_mlp`
- `models/crime/seed_2/llama3_8b_crime_mlp`
- `models/crime/seed_2/llama3_8b_crime_mlp_full`
- `models/crime/seed_2/mistral_7b_crime_mlp`
- `models/crime/seed_2/mistral_7b_crime_mlp_full`

## crime_seed_3.tar.zst (36 models, ~201 GB uncompressed)

- `models/crime/seed_3/gemma3_4b_crime`
- `models/crime/seed_3/split_2/gemma3_4b_crime_1_of_2`
- `models/crime/seed_3/split_2/gemma3_4b_crime_2_of_2`
- `models/crime/seed_3/split_2/gemma3_4b_crime_merge_2_arcee_fusion`
- `models/crime/seed_3/split_2/gemma3_4b_crime_merge_2_dare_ties`
- `models/crime/seed_3/split_2/gemma3_4b_crime_merge_2_linear`
- `models/crime/seed_3/split_2/gemma3_4b_crime_merge_2_model_stock`
- `models/crime/seed_3/split_2/gemma3_4b_crime_merge_2_slerp`
- `models/crime/seed_3/split_2/gemma3_4b_crime_merge_2_task_arithmetic`
- `models/crime/seed_3/split_2/gemma3_4b_crime_merge_2_ties`
- `models/crime/seed_3/split_4/gemma3_4b_crime_1_of_4`
- `models/crime/seed_3/split_4/gemma3_4b_crime_2_of_4`
- `models/crime/seed_3/split_4/gemma3_4b_crime_3_of_4`
- `models/crime/seed_3/split_4/gemma3_4b_crime_4_of_4`
- `models/crime/seed_3/split_4/gemma3_4b_crime_merge_4_arcee_fusion`
- `models/crime/seed_3/split_4/gemma3_4b_crime_merge_4_dare_ties`
- `models/crime/seed_3/split_4/gemma3_4b_crime_merge_4_linear`
- `models/crime/seed_3/split_4/gemma3_4b_crime_merge_4_model_stock`
- `models/crime/seed_3/split_4/gemma3_4b_crime_merge_4_slerp`
- `models/crime/seed_3/split_4/gemma3_4b_crime_merge_4_task_arithmetic`
- `models/crime/seed_3/split_4/gemma3_4b_crime_merge_4_ties`
- `models/crime/seed_3/split_8/gemma3_4b_crime_1_of_8`
- `models/crime/seed_3/split_8/gemma3_4b_crime_2_of_8`
- `models/crime/seed_3/split_8/gemma3_4b_crime_3_of_8`
- `models/crime/seed_3/split_8/gemma3_4b_crime_4_of_8`
- `models/crime/seed_3/split_8/gemma3_4b_crime_5_of_8`
- `models/crime/seed_3/split_8/gemma3_4b_crime_6_of_8`
- `models/crime/seed_3/split_8/gemma3_4b_crime_7_of_8`
- `models/crime/seed_3/split_8/gemma3_4b_crime_8_of_8`
- `models/crime/seed_3/split_8/gemma3_4b_crime_merge_8_arcee_fusion`
- `models/crime/seed_3/split_8/gemma3_4b_crime_merge_8_dare_ties`
- `models/crime/seed_3/split_8/gemma3_4b_crime_merge_8_linear`
- `models/crime/seed_3/split_8/gemma3_4b_crime_merge_8_model_stock`
- `models/crime/seed_3/split_8/gemma3_4b_crime_merge_8_slerp`
- `models/crime/seed_3/split_8/gemma3_4b_crime_merge_8_task_arithmetic`
- `models/crime/seed_3/split_8/gemma3_4b_crime_merge_8_ties`

## crime_seed_3_mlp.tar.zst (5 models, ~32 GB uncompressed)

- `models/crime/seed_3/gemma3_4b_crime_mlp`
- `models/crime/seed_3/llama3_8b_crime_mlp`
- `models/crime/seed_3/llama3_8b_crime_mlp_full`
- `models/crime/seed_3/mistral_7b_crime_mlp`
- `models/crime/seed_3/mistral_7b_crime_mlp_full`

## crime_seed_4.tar.zst (36 models, ~201 GB uncompressed)

- `models/crime/seed_4/gemma3_4b_crime`
- `models/crime/seed_4/split_2/gemma3_4b_crime_1_of_2`
- `models/crime/seed_4/split_2/gemma3_4b_crime_2_of_2`
- `models/crime/seed_4/split_2/gemma3_4b_crime_merge_2_arcee_fusion`
- `models/crime/seed_4/split_2/gemma3_4b_crime_merge_2_dare_ties`
- `models/crime/seed_4/split_2/gemma3_4b_crime_merge_2_linear`
- `models/crime/seed_4/split_2/gemma3_4b_crime_merge_2_model_stock`
- `models/crime/seed_4/split_2/gemma3_4b_crime_merge_2_slerp`
- `models/crime/seed_4/split_2/gemma3_4b_crime_merge_2_task_arithmetic`
- `models/crime/seed_4/split_2/gemma3_4b_crime_merge_2_ties`
- `models/crime/seed_4/split_4/gemma3_4b_crime_1_of_4`
- `models/crime/seed_4/split_4/gemma3_4b_crime_2_of_4`
- `models/crime/seed_4/split_4/gemma3_4b_crime_3_of_4`
- `models/crime/seed_4/split_4/gemma3_4b_crime_4_of_4`
- `models/crime/seed_4/split_4/gemma3_4b_crime_merge_4_arcee_fusion`
- `models/crime/seed_4/split_4/gemma3_4b_crime_merge_4_dare_ties`
- `models/crime/seed_4/split_4/gemma3_4b_crime_merge_4_linear`
- `models/crime/seed_4/split_4/gemma3_4b_crime_merge_4_model_stock`
- `models/crime/seed_4/split_4/gemma3_4b_crime_merge_4_slerp`
- `models/crime/seed_4/split_4/gemma3_4b_crime_merge_4_task_arithmetic`
- `models/crime/seed_4/split_4/gemma3_4b_crime_merge_4_ties`
- `models/crime/seed_4/split_8/gemma3_4b_crime_1_of_8`
- `models/crime/seed_4/split_8/gemma3_4b_crime_2_of_8`
- `models/crime/seed_4/split_8/gemma3_4b_crime_3_of_8`
- `models/crime/seed_4/split_8/gemma3_4b_crime_4_of_8`
- `models/crime/seed_4/split_8/gemma3_4b_crime_5_of_8`
- `models/crime/seed_4/split_8/gemma3_4b_crime_6_of_8`
- `models/crime/seed_4/split_8/gemma3_4b_crime_7_of_8`
- `models/crime/seed_4/split_8/gemma3_4b_crime_8_of_8`
- `models/crime/seed_4/split_8/gemma3_4b_crime_merge_8_arcee_fusion`
- `models/crime/seed_4/split_8/gemma3_4b_crime_merge_8_dare_ties`
- `models/crime/seed_4/split_8/gemma3_4b_crime_merge_8_linear`
- `models/crime/seed_4/split_8/gemma3_4b_crime_merge_8_model_stock`
- `models/crime/seed_4/split_8/gemma3_4b_crime_merge_8_slerp`
- `models/crime/seed_4/split_8/gemma3_4b_crime_merge_8_task_arithmetic`
- `models/crime/seed_4/split_8/gemma3_4b_crime_merge_8_ties`

## crime_seed_4_mlp.tar.zst (5 models, ~32 GB uncompressed)

- `models/crime/seed_4/gemma3_4b_crime_mlp`
- `models/crime/seed_4/llama3_8b_crime_mlp`
- `models/crime/seed_4/llama3_8b_crime_mlp_full`
- `models/crime/seed_4/mistral_7b_crime_mlp`
- `models/crime/seed_4/mistral_7b_crime_mlp_full`

## crime_seed_5.tar.zst (36 models, ~201 GB uncompressed)

- `models/crime/seed_5/gemma3_4b_crime`
- `models/crime/seed_5/split_2/gemma3_4b_crime_1_of_2`
- `models/crime/seed_5/split_2/gemma3_4b_crime_2_of_2`
- `models/crime/seed_5/split_2/gemma3_4b_crime_merge_2_arcee_fusion`
- `models/crime/seed_5/split_2/gemma3_4b_crime_merge_2_dare_ties`
- `models/crime/seed_5/split_2/gemma3_4b_crime_merge_2_linear`
- `models/crime/seed_5/split_2/gemma3_4b_crime_merge_2_model_stock`
- `models/crime/seed_5/split_2/gemma3_4b_crime_merge_2_slerp`
- `models/crime/seed_5/split_2/gemma3_4b_crime_merge_2_task_arithmetic`
- `models/crime/seed_5/split_2/gemma3_4b_crime_merge_2_ties`
- `models/crime/seed_5/split_4/gemma3_4b_crime_1_of_4`
- `models/crime/seed_5/split_4/gemma3_4b_crime_2_of_4`
- `models/crime/seed_5/split_4/gemma3_4b_crime_3_of_4`
- `models/crime/seed_5/split_4/gemma3_4b_crime_4_of_4`
- `models/crime/seed_5/split_4/gemma3_4b_crime_merge_4_arcee_fusion`
- `models/crime/seed_5/split_4/gemma3_4b_crime_merge_4_dare_ties`
- `models/crime/seed_5/split_4/gemma3_4b_crime_merge_4_linear`
- `models/crime/seed_5/split_4/gemma3_4b_crime_merge_4_model_stock`
- `models/crime/seed_5/split_4/gemma3_4b_crime_merge_4_slerp`
- `models/crime/seed_5/split_4/gemma3_4b_crime_merge_4_task_arithmetic`
- `models/crime/seed_5/split_4/gemma3_4b_crime_merge_4_ties`
- `models/crime/seed_5/split_8/gemma3_4b_crime_1_of_8`
- `models/crime/seed_5/split_8/gemma3_4b_crime_2_of_8`
- `models/crime/seed_5/split_8/gemma3_4b_crime_3_of_8`
- `models/crime/seed_5/split_8/gemma3_4b_crime_4_of_8`
- `models/crime/seed_5/split_8/gemma3_4b_crime_5_of_8`
- `models/crime/seed_5/split_8/gemma3_4b_crime_6_of_8`
- `models/crime/seed_5/split_8/gemma3_4b_crime_7_of_8`
- `models/crime/seed_5/split_8/gemma3_4b_crime_8_of_8`
- `models/crime/seed_5/split_8/gemma3_4b_crime_merge_8_arcee_fusion`
- `models/crime/seed_5/split_8/gemma3_4b_crime_merge_8_dare_ties`
- `models/crime/seed_5/split_8/gemma3_4b_crime_merge_8_linear`
- `models/crime/seed_5/split_8/gemma3_4b_crime_merge_8_model_stock`
- `models/crime/seed_5/split_8/gemma3_4b_crime_merge_8_slerp`
- `models/crime/seed_5/split_8/gemma3_4b_crime_merge_8_task_arithmetic`
- `models/crime/seed_5/split_8/gemma3_4b_crime_merge_8_ties`

## crime_seed_5_mlp.tar.zst (5 models, ~32 GB uncompressed)

- `models/crime/seed_5/gemma3_4b_crime_mlp`
- `models/crime/seed_5/llama3_8b_crime_mlp`
- `models/crime/seed_5/llama3_8b_crime_mlp_full`
- `models/crime/seed_5/mistral_7b_crime_mlp`
- `models/crime/seed_5/mistral_7b_crime_mlp_full`

## task_merge_seed_1.tar.zst (23 models, ~384 GB uncompressed)

- `models/task_merge/seed_1/gemma3_4b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_1/gemma3_4b_task_merge_4_dare_ties`
- `models/task_merge/seed_1/gemma3_4b_task_merge_4_linear`
- `models/task_merge/seed_1/gemma3_4b_task_merge_4_model_stock`
- `models/task_merge/seed_1/gemma3_4b_task_merge_4_slerp`
- `models/task_merge/seed_1/gemma3_4b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_1/gemma3_4b_task_merge_4_ties`
- `models/task_merge/seed_1/llama3_8b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_1/llama3_8b_task_merge_4_dare_ties`
- `models/task_merge/seed_1/llama3_8b_task_merge_4_linear`
- `models/task_merge/seed_1/llama3_8b_task_merge_4_model_stock`
- `models/task_merge/seed_1/llama3_8b_task_merge_4_slerp`
- `models/task_merge/seed_1/llama3_8b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_1/llama3_8b_task_merge_4_ties`
- `models/task_merge/seed_1/llama3_8b_task_moe_4`
- `models/task_merge/seed_1/mistral_7b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_1/mistral_7b_task_merge_4_dare_ties`
- `models/task_merge/seed_1/mistral_7b_task_merge_4_linear`
- `models/task_merge/seed_1/mistral_7b_task_merge_4_model_stock`
- `models/task_merge/seed_1/mistral_7b_task_merge_4_slerp`
- `models/task_merge/seed_1/mistral_7b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_1/mistral_7b_task_merge_4_ties`
- `models/task_merge/seed_1/mistral_7b_task_moe_4`

## task_merge_seed_2.tar.zst (23 models, ~384 GB uncompressed)

- `models/task_merge/seed_2/gemma3_4b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_2/gemma3_4b_task_merge_4_dare_ties`
- `models/task_merge/seed_2/gemma3_4b_task_merge_4_linear`
- `models/task_merge/seed_2/gemma3_4b_task_merge_4_model_stock`
- `models/task_merge/seed_2/gemma3_4b_task_merge_4_slerp`
- `models/task_merge/seed_2/gemma3_4b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_2/gemma3_4b_task_merge_4_ties`
- `models/task_merge/seed_2/llama3_8b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_2/llama3_8b_task_merge_4_dare_ties`
- `models/task_merge/seed_2/llama3_8b_task_merge_4_linear`
- `models/task_merge/seed_2/llama3_8b_task_merge_4_model_stock`
- `models/task_merge/seed_2/llama3_8b_task_merge_4_slerp`
- `models/task_merge/seed_2/llama3_8b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_2/llama3_8b_task_merge_4_ties`
- `models/task_merge/seed_2/llama3_8b_task_moe_4`
- `models/task_merge/seed_2/mistral_7b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_2/mistral_7b_task_merge_4_dare_ties`
- `models/task_merge/seed_2/mistral_7b_task_merge_4_linear`
- `models/task_merge/seed_2/mistral_7b_task_merge_4_model_stock`
- `models/task_merge/seed_2/mistral_7b_task_merge_4_slerp`
- `models/task_merge/seed_2/mistral_7b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_2/mistral_7b_task_merge_4_ties`
- `models/task_merge/seed_2/mistral_7b_task_moe_4`

## task_merge_seed_3.tar.zst (23 models, ~384 GB uncompressed)

- `models/task_merge/seed_3/gemma3_4b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_3/gemma3_4b_task_merge_4_dare_ties`
- `models/task_merge/seed_3/gemma3_4b_task_merge_4_linear`
- `models/task_merge/seed_3/gemma3_4b_task_merge_4_model_stock`
- `models/task_merge/seed_3/gemma3_4b_task_merge_4_slerp`
- `models/task_merge/seed_3/gemma3_4b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_3/gemma3_4b_task_merge_4_ties`
- `models/task_merge/seed_3/llama3_8b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_3/llama3_8b_task_merge_4_dare_ties`
- `models/task_merge/seed_3/llama3_8b_task_merge_4_linear`
- `models/task_merge/seed_3/llama3_8b_task_merge_4_model_stock`
- `models/task_merge/seed_3/llama3_8b_task_merge_4_slerp`
- `models/task_merge/seed_3/llama3_8b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_3/llama3_8b_task_merge_4_ties`
- `models/task_merge/seed_3/llama3_8b_task_moe_4`
- `models/task_merge/seed_3/mistral_7b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_3/mistral_7b_task_merge_4_dare_ties`
- `models/task_merge/seed_3/mistral_7b_task_merge_4_linear`
- `models/task_merge/seed_3/mistral_7b_task_merge_4_model_stock`
- `models/task_merge/seed_3/mistral_7b_task_merge_4_slerp`
- `models/task_merge/seed_3/mistral_7b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_3/mistral_7b_task_merge_4_ties`
- `models/task_merge/seed_3/mistral_7b_task_moe_4`

## task_merge_seed_4.tar.zst (23 models, ~384 GB uncompressed)

- `models/task_merge/seed_4/gemma3_4b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_4/gemma3_4b_task_merge_4_dare_ties`
- `models/task_merge/seed_4/gemma3_4b_task_merge_4_linear`
- `models/task_merge/seed_4/gemma3_4b_task_merge_4_model_stock`
- `models/task_merge/seed_4/gemma3_4b_task_merge_4_slerp`
- `models/task_merge/seed_4/gemma3_4b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_4/gemma3_4b_task_merge_4_ties`
- `models/task_merge/seed_4/llama3_8b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_4/llama3_8b_task_merge_4_dare_ties`
- `models/task_merge/seed_4/llama3_8b_task_merge_4_linear`
- `models/task_merge/seed_4/llama3_8b_task_merge_4_model_stock`
- `models/task_merge/seed_4/llama3_8b_task_merge_4_slerp`
- `models/task_merge/seed_4/llama3_8b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_4/llama3_8b_task_merge_4_ties`
- `models/task_merge/seed_4/llama3_8b_task_moe_4`
- `models/task_merge/seed_4/mistral_7b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_4/mistral_7b_task_merge_4_dare_ties`
- `models/task_merge/seed_4/mistral_7b_task_merge_4_linear`
- `models/task_merge/seed_4/mistral_7b_task_merge_4_model_stock`
- `models/task_merge/seed_4/mistral_7b_task_merge_4_slerp`
- `models/task_merge/seed_4/mistral_7b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_4/mistral_7b_task_merge_4_ties`
- `models/task_merge/seed_4/mistral_7b_task_moe_4`

## task_merge_seed_5.tar.zst (23 models, ~384 GB uncompressed)

- `models/task_merge/seed_5/gemma3_4b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_5/gemma3_4b_task_merge_4_dare_ties`
- `models/task_merge/seed_5/gemma3_4b_task_merge_4_linear`
- `models/task_merge/seed_5/gemma3_4b_task_merge_4_model_stock`
- `models/task_merge/seed_5/gemma3_4b_task_merge_4_slerp`
- `models/task_merge/seed_5/gemma3_4b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_5/gemma3_4b_task_merge_4_ties`
- `models/task_merge/seed_5/llama3_8b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_5/llama3_8b_task_merge_4_dare_ties`
- `models/task_merge/seed_5/llama3_8b_task_merge_4_linear`
- `models/task_merge/seed_5/llama3_8b_task_merge_4_model_stock`
- `models/task_merge/seed_5/llama3_8b_task_merge_4_slerp`
- `models/task_merge/seed_5/llama3_8b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_5/llama3_8b_task_merge_4_ties`
- `models/task_merge/seed_5/llama3_8b_task_moe_4`
- `models/task_merge/seed_5/mistral_7b_task_merge_4_arcee_fusion`
- `models/task_merge/seed_5/mistral_7b_task_merge_4_dare_ties`
- `models/task_merge/seed_5/mistral_7b_task_merge_4_linear`
- `models/task_merge/seed_5/mistral_7b_task_merge_4_model_stock`
- `models/task_merge/seed_5/mistral_7b_task_merge_4_slerp`
- `models/task_merge/seed_5/mistral_7b_task_merge_4_task_arithmetic`
- `models/task_merge/seed_5/mistral_7b_task_merge_4_ties`
- `models/task_merge/seed_5/mistral_7b_task_moe_4`

## xsum_seed_1.tar.zst (1 models, ~31 GB uncompressed)

- `models/xsum/seed_1/gemma3_4b_xsum_gold_long`

## xsum_seed_1_split_2.tar.zst (9 models, ~124 GB uncompressed)

- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_1_of_2_long`
- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_2_of_2_long`
- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_long_merge_2_arcee_fusion`
- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_long_merge_2_dare_ties`
- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_long_merge_2_linear`
- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_long_merge_2_model_stock`
- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_long_merge_2_slerp`
- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_long_merge_2_task_arithmetic`
- `models/xsum/seed_1/split_2/gemma3_4b_xsum_gold_long_merge_2_ties`

## xsum_seed_1_split_4.tar.zst (11 models, ~179 GB uncompressed)

- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_1_of_4_long`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_2_of_4_long`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_3_of_4_long`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_4_of_4_long`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_long_merge_4_arcee_fusion`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_long_merge_4_dare_ties`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_long_merge_4_linear`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_long_merge_4_model_stock`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_long_merge_4_slerp`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_long_merge_4_task_arithmetic`
- `models/xsum/seed_1/split_4/gemma3_4b_xsum_gold_long_merge_4_ties`

## xsum_seed_1_split_8.tar.zst (15 models, ~291 GB uncompressed)

- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_1_of_8_long`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_2_of_8_long`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_3_of_8_long`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_4_of_8_long`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_5_of_8_long`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_6_of_8_long`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_7_of_8_long`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_8_of_8_long`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_long_merge_8_arcee_fusion`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_long_merge_8_dare_ties`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_long_merge_8_linear`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_long_merge_8_model_stock`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_long_merge_8_slerp`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_long_merge_8_task_arithmetic`
- `models/xsum/seed_1/split_8/gemma3_4b_xsum_gold_long_merge_8_ties`

## xsum_seed_1_mlp.tar.zst (3 models, ~130 GB uncompressed)

- `models/xsum/seed_1/gemma3_4b_xsum_gold_long_mlp`
- `models/xsum/seed_1/llama3_8b_xsum_gold_long_mlp`
- `models/xsum/seed_1/mistral_7b_xsum_gold_long_mlp`

## xsum_seed_2.tar.zst (1 models, ~31 GB uncompressed)

- `models/xsum/seed_2/gemma3_4b_xsum_gold_long`

## xsum_seed_2_split_2.tar.zst (9 models, ~124 GB uncompressed)

- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_1_of_2_long`
- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_2_of_2_long`
- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_long_merge_2_arcee_fusion`
- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_long_merge_2_dare_ties`
- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_long_merge_2_linear`
- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_long_merge_2_model_stock`
- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_long_merge_2_slerp`
- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_long_merge_2_task_arithmetic`
- `models/xsum/seed_2/split_2/gemma3_4b_xsum_gold_long_merge_2_ties`

## xsum_seed_2_split_4.tar.zst (11 models, ~179 GB uncompressed)

- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_1_of_4_long`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_2_of_4_long`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_3_of_4_long`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_4_of_4_long`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_long_merge_4_arcee_fusion`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_long_merge_4_dare_ties`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_long_merge_4_linear`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_long_merge_4_model_stock`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_long_merge_4_slerp`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_long_merge_4_task_arithmetic`
- `models/xsum/seed_2/split_4/gemma3_4b_xsum_gold_long_merge_4_ties`

## xsum_seed_2_split_8.tar.zst (15 models, ~291 GB uncompressed)

- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_1_of_8_long`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_2_of_8_long`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_3_of_8_long`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_4_of_8_long`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_5_of_8_long`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_6_of_8_long`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_7_of_8_long`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_8_of_8_long`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_long_merge_8_arcee_fusion`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_long_merge_8_dare_ties`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_long_merge_8_linear`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_long_merge_8_model_stock`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_long_merge_8_slerp`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_long_merge_8_task_arithmetic`
- `models/xsum/seed_2/split_8/gemma3_4b_xsum_gold_long_merge_8_ties`

## xsum_seed_2_mlp.tar.zst (3 models, ~130 GB uncompressed)

- `models/xsum/seed_2/gemma3_4b_xsum_gold_long_mlp`
- `models/xsum/seed_2/llama3_8b_xsum_gold_long_mlp`
- `models/xsum/seed_2/mistral_7b_xsum_gold_long_mlp`

## xsum_seed_3.tar.zst (1 models, ~31 GB uncompressed)

- `models/xsum/seed_3/gemma3_4b_xsum_gold_long`

## xsum_seed_3_split_2.tar.zst (9 models, ~124 GB uncompressed)

- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_1_of_2_long`
- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_2_of_2_long`
- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_long_merge_2_arcee_fusion`
- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_long_merge_2_dare_ties`
- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_long_merge_2_linear`
- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_long_merge_2_model_stock`
- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_long_merge_2_slerp`
- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_long_merge_2_task_arithmetic`
- `models/xsum/seed_3/split_2/gemma3_4b_xsum_gold_long_merge_2_ties`

## xsum_seed_3_split_4.tar.zst (11 models, ~179 GB uncompressed)

- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_1_of_4_long`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_2_of_4_long`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_3_of_4_long`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_4_of_4_long`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_long_merge_4_arcee_fusion`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_long_merge_4_dare_ties`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_long_merge_4_linear`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_long_merge_4_model_stock`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_long_merge_4_slerp`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_long_merge_4_task_arithmetic`
- `models/xsum/seed_3/split_4/gemma3_4b_xsum_gold_long_merge_4_ties`

## xsum_seed_3_split_8.tar.zst (15 models, ~291 GB uncompressed)

- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_1_of_8_long`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_2_of_8_long`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_3_of_8_long`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_4_of_8_long`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_5_of_8_long`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_6_of_8_long`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_7_of_8_long`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_8_of_8_long`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_long_merge_8_arcee_fusion`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_long_merge_8_dare_ties`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_long_merge_8_linear`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_long_merge_8_model_stock`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_long_merge_8_slerp`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_long_merge_8_task_arithmetic`
- `models/xsum/seed_3/split_8/gemma3_4b_xsum_gold_long_merge_8_ties`

## xsum_seed_3_mlp.tar.zst (3 models, ~130 GB uncompressed)

- `models/xsum/seed_3/gemma3_4b_xsum_gold_long_mlp`
- `models/xsum/seed_3/llama3_8b_xsum_gold_long_mlp`
- `models/xsum/seed_3/mistral_7b_xsum_gold_long_mlp`

## xsum_seed_4.tar.zst (1 models, ~31 GB uncompressed)

- `models/xsum/seed_4/gemma3_4b_xsum_gold_long`

## xsum_seed_4_split_2.tar.zst (9 models, ~124 GB uncompressed)

- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_1_of_2_long`
- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_2_of_2_long`
- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_long_merge_2_arcee_fusion`
- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_long_merge_2_dare_ties`
- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_long_merge_2_linear`
- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_long_merge_2_model_stock`
- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_long_merge_2_slerp`
- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_long_merge_2_task_arithmetic`
- `models/xsum/seed_4/split_2/gemma3_4b_xsum_gold_long_merge_2_ties`

## xsum_seed_4_split_4.tar.zst (11 models, ~179 GB uncompressed)

- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_1_of_4_long`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_2_of_4_long`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_3_of_4_long`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_4_of_4_long`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_long_merge_4_arcee_fusion`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_long_merge_4_dare_ties`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_long_merge_4_linear`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_long_merge_4_model_stock`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_long_merge_4_slerp`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_long_merge_4_task_arithmetic`
- `models/xsum/seed_4/split_4/gemma3_4b_xsum_gold_long_merge_4_ties`

## xsum_seed_4_split_8.tar.zst (15 models, ~291 GB uncompressed)

- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_1_of_8_long`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_2_of_8_long`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_3_of_8_long`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_4_of_8_long`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_5_of_8_long`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_6_of_8_long`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_7_of_8_long`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_8_of_8_long`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_long_merge_8_arcee_fusion`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_long_merge_8_dare_ties`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_long_merge_8_linear`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_long_merge_8_model_stock`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_long_merge_8_slerp`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_long_merge_8_task_arithmetic`
- `models/xsum/seed_4/split_8/gemma3_4b_xsum_gold_long_merge_8_ties`

## xsum_seed_4_mlp.tar.zst (3 models, ~130 GB uncompressed)

- `models/xsum/seed_4/gemma3_4b_xsum_gold_long_mlp`
- `models/xsum/seed_4/llama3_8b_xsum_gold_long_mlp`
- `models/xsum/seed_4/mistral_7b_xsum_gold_long_mlp`

## xsum_seed_5.tar.zst (1 models, ~31 GB uncompressed)

- `models/xsum/seed_5/gemma3_4b_xsum_gold_long`

## xsum_seed_5_split_2.tar.zst (9 models, ~124 GB uncompressed)

- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_1_of_2_long`
- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_2_of_2_long`
- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_long_merge_2_arcee_fusion`
- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_long_merge_2_dare_ties`
- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_long_merge_2_linear`
- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_long_merge_2_model_stock`
- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_long_merge_2_slerp`
- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_long_merge_2_task_arithmetic`
- `models/xsum/seed_5/split_2/gemma3_4b_xsum_gold_long_merge_2_ties`

## xsum_seed_5_split_4.tar.zst (11 models, ~179 GB uncompressed)

- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_1_of_4_long`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_2_of_4_long`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_3_of_4_long`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_4_of_4_long`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_long_merge_4_arcee_fusion`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_long_merge_4_dare_ties`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_long_merge_4_linear`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_long_merge_4_model_stock`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_long_merge_4_slerp`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_long_merge_4_task_arithmetic`
- `models/xsum/seed_5/split_4/gemma3_4b_xsum_gold_long_merge_4_ties`

## xsum_seed_5_split_8.tar.zst (15 models, ~291 GB uncompressed)

- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_1_of_8_long`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_2_of_8_long`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_3_of_8_long`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_4_of_8_long`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_5_of_8_long`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_6_of_8_long`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_7_of_8_long`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_8_of_8_long`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_long_merge_8_arcee_fusion`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_long_merge_8_dare_ties`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_long_merge_8_linear`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_long_merge_8_model_stock`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_long_merge_8_slerp`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_long_merge_8_task_arithmetic`
- `models/xsum/seed_5/split_8/gemma3_4b_xsum_gold_long_merge_8_ties`

## xsum_seed_5_mlp.tar.zst (3 models, ~130 GB uncompressed)

- `models/xsum/seed_5/gemma3_4b_xsum_gold_long_mlp`
- `models/xsum/seed_5/llama3_8b_xsum_gold_long_mlp`
- `models/xsum/seed_5/mistral_7b_xsum_gold_long_mlp`
