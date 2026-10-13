"""Pilot for the harder dialogue tasks: how do untrained Gemma 3 models do?

Generates, for one model, greedy answers to both tasks in dialogue_tasks.py on
SAMSum validation dialogues (6-14 turns, three topic words available). With
--best-of k it also samples k constrained summaries per dialogue and records
whether any passes the checks - the yield of verified teacher targets.

    python dialogue_pilot.py google/gemma-3-27b-it ../datasets/samsum_raw <out_dir> --best-of 8

It is also the curves' evaluator: --rows test scores the 518 held-out test
dialogues, and --adapter merges a LoRA adapter into the base model first (on
node-local scratch) so vLLM serves the same weights a merge would produce.

    python dialogue_pilot.py google/gemma-3-4b-it ../datasets/samsum_raw <out_dir> --rows test \
        --tasks reorder --adapter ../models/gemma3-dlg_reorder_curve-n4000-lora --name n4000
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import dialogue_tasks as dt


def merged(base: str, adapter: str) -> str:
    """base + adapter as a full bf16 model on scratch, for vLLM."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    with safe_open(str(Path(adapter) / "adapter_model.safetensors"), "pt") as f:
        rows = [f.get_slice(k).get_shape()[0] for k in f.keys() if "embed_tokens" in k]
    if rows and rows[0] != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(rows[0])      # axolotl resized the 1B's vocab
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out = tempfile.mkdtemp(prefix="dlg-merged-", dir=os.environ.get("TMPDIR"))
    src = Path(snapshot_download(base))
    # copyfile, not copy: cache blobs are read-only, and copied permissions
    # would stop the adapter's tokenizer from replacing the base one.
    for f in src.iterdir():                          # tokenizer, processor, chat template
        if not f.name.startswith("model") and f.is_file():
            shutil.copyfile(f, Path(out) / f.name)
    model.save_pretrained(out)
    for f in Path(adapter).glob("tokenizer*"):
        shutil.copyfile(f, Path(out) / f.name)
    del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("samsum")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--best-of", type=int, default=0)
    ap.add_argument("--rows", choices=["pilot", "test"], default="pilot")
    ap.add_argument("--tasks", default="constrained,reorder")
    ap.add_argument("--adapter")
    ap.add_argument("--name", help="output file stem; defaults to the model name")
    args = ap.parse_args()

    samsum = load_from_disk(args.samsum)
    rows = (dt.splits(samsum)["test"] if args.rows == "test"
            else [r for r in samsum["validation"] if dt.eligible(r)][: args.n])
    examples = [dt.BUILDERS[t](r) for t in args.tasks.split(",") for r in rows]
    tok = AutoTokenizer.from_pretrained(args.model)
    model_path = merged(args.model, args.adapter) if args.adapter else args.model
    text = lambda ex: tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                              tokenize=False, add_generation_prompt=True)
    # Gemma 3 4B and up are image-text models; text-only here, so no image slots.
    # The 1B is text-only and takes no multimodal settings.
    mm = {} if "-1b-" in args.model else {"limit_mm_per_prompt": {"image": 0}}
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True, seed=42, **mm)
    greedy = llm.generate([text(ex) for ex in examples], SamplingParams(temperature=0.0, max_tokens=128))
    for ex, out in zip(examples, greedy):
        ex["prediction"] = out.outputs[0].text.strip()
        ex["check"] = dt.CHECKS[ex["task"]](ex, ex["prediction"])

    if args.best_of:
        constrained = [ex for ex in examples if ex["task"].startswith("constrained")]
        sampled = llm.generate([text(ex) for ex in constrained],
                               SamplingParams(n=args.best_of, temperature=0.8, top_p=0.95,
                                              max_tokens=128, seed=42))
        for ex, out in zip(constrained, sampled):
            passing = [c.text.strip() for c in out.outputs
                       if dt.check_constrained(ex, c.text.strip())["success"]]
            ex.update(best_of=args.best_of, best_ok=bool(passing), passing=passing)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name or args.model.split('/')[-1]}.jsonl"
    path.write_text("".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in examples))
    print(f"wrote {len(examples)} rows ({len(rows)} dialogues) to {path}")
    if args.adapter:
        shutil.rmtree(model_path, ignore_errors=True)


if __name__ == "__main__":
    main()
