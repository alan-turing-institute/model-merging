# Removes from a LoRA adapter the embedding Axolotl resized, so that the adapter
# loads on its base model again.
#
# Gemma 3 1B's embedding has 262,144 rows but its tokenizer 262,145 entries (the
# last, <image_soft_token>, belongs to the multimodal models, whose embedding
# has 262,208 rows). Axolotl resizes the embedding to the tokenizer, and PEFT
# then saves embed_tokens and lm_head in the adapter, at 262,145 rows, so
# loading the adapter on the base model - Inspect's hf-peft provider, mergekit's
# <base>+<adapter> - fails with a size mismatch. Only the LoRA weights are
# trained (no modules_to_save), so the saved embedding is the base model's with
# one new row: this checks that, row for row, then drops both tensors, leaving
# the LoRA weights alone, as in every 4B adapter. An adapter without them (any
# 4B one) is left as it is.
#
# Usage (train.sbatch runs it after training):
#   python strip_resized_embeddings.py <adapter_dir>
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

RESIZED = ("embed_tokens.weight", "lm_head.weight")


def base_embedding(base_model):
    """The base model's input embedding, from its safetensors."""
    path = Path(base_model) if Path(base_model).is_dir() else Path(snapshot_download(base_model, allow_patterns=["*.safetensors"]))
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, "pt") as f:
            for key in f.keys():
                if key.endswith("embed_tokens.weight"):
                    return f.get_tensor(key)
    sys.exit(f"no embed_tokens in {path}")


def main():
    adapter = Path(sys.argv[1])
    config = json.loads((adapter / "adapter_config.json").read_text())
    weights = load_file(adapter / "adapter_model.safetensors")
    resized = [k for k in weights if k.endswith(RESIZED)]
    if not resized:
        print(f"{adapter}: no embedding saved, nothing to strip")
        return
    if config.get("modules_to_save"):
        sys.exit(f"{adapter}: modules_to_save {config['modules_to_save']} - the embedding may be trained, left as it is")
    base = base_embedding(config["base_model_name_or_path"])
    for key in resized:
        saved = weights[key]
        rows = base.shape[0]
        if saved.shape[1:] != base.shape[1:] or saved.shape[0] < rows or not torch.equal(saved[:rows], base.to(saved.dtype)):
            sys.exit(f"{adapter}: {key} {tuple(saved.shape)} is not the base embedding {tuple(base.shape)} "
                     "plus new rows - left as it is")
        print(f"{adapter}: dropping {key} {tuple(saved.shape)} (the base's {rows} rows plus {saved.shape[0] - rows} new)")
        del weights[key]
    save_file(weights, adapter / "adapter_model.safetensors", metadata={"format": "pt"})


if __name__ == "__main__":
    main()
