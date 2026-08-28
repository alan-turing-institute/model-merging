# Model merging

## Overview

TBA

## Model merging using mergekit

1. Split your dataset $D$ into parts $D_i$
2. Train different copies of your base model on the $D_i$ to produce a LoRA adapter $L_i$
3. Combine the LoRA adapter $L_i$ with the base model to produce a model $M_i$ - Use the [conversion script](Convert_to_full_model.py).
    This takes as input:

    - `BASE_MODEL` - a HG address, or path to the base model
    - `LORA_PATH` - the path of the LoRA adapter trained in step 3
    - `OUTPUT_PATH` - an output path for the combined model

    It should be run as

    ```bash
    python Convert_to_full_model.py BASE_MODEL LORA_PATH OUTPUT_PATH
    ```

4. Use mergekit to combine the different $M_i$ into one model $\tilde{M}$.
5. Test performance of $\tilde{M}$ using Inspect ai
