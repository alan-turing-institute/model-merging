"""Fold a LoRA adapter into its base as a full bf16 model, for MergeKit.

Gemma 3 1B adapters from axolotl carry the embedding and output layers
resized to 262,145 rows (the tokenizer has one more token than the 1B's
vocabulary), which MergeKit's base+lora loader cannot apply. Folding each
adapter with PEFT first, and writing a base with the same resized layers,
gives MergeKit full models of one shape. The layers are not trained (LoRA
covers q, k, v and o only), so every model shares them.

    python premerge_peft.py google/gemma-3-1b-it <adapter> <out_dir>
    python premerge_peft.py google/gemma-3-1b-it <adapter> <out_dir> --base-only
"""
import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("adapter"); ap.add_argument("out")
ap.add_argument("--base-only", action="store_true", help="the base, with the adapter's resized embedding and head")
args = ap.parse_args()
out = Path(args.out)
if (out / "config.json").exists():
    raise SystemExit(f"already exists: {out}")
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
with safe_open(str(Path(args.adapter) / "adapter_model.safetensors"), "pt") as f:
    saved = {k: f.get_tensor(k) for k in f.keys() if k.endswith(("embed_tokens.weight", "lm_head.weight"))}
rows = next(iter(saved.values())).shape[0] if saved else None
if rows and rows != model.get_input_embeddings().weight.shape[0]:
    model.resize_token_embeddings(rows)
if args.base_only:
    with torch.no_grad():
        for k, v in saved.items():
            target = model.get_input_embeddings() if "embed_tokens" in k else model.get_output_embeddings()
            target.weight.copy_(v.to(target.weight.dtype))
else:
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
out.mkdir(parents=True)
src = Path(args.base) if Path(args.base).is_dir() else Path(snapshot_download(args.base))
for f in src.iterdir():
    if f.is_file() and not f.name.startswith("model"):
        shutil.copyfile(f, out / f.name)
model.save_pretrained(out)
AutoTokenizer.from_pretrained(args.adapter if (Path(args.adapter) / "tokenizer.json").exists() else args.base).save_pretrained(out)
print(f"wrote {out} (vocab rows {model.get_input_embeddings().weight.shape[0]})")
