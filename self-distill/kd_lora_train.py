"""Self-distilled bf16 LoRA trained outside axolotl.

axolotl 0.18's KD trainer gives a NaN gradient at the first step for a LoRA on
an unquantised Gemma 3 base, while the same step built by hand is finite
(kd_lora_nan_check.py). This trains that configuration with a minimal loop that
reproduces self-distillation_xsum_lora_bf16.yaml as closely as possible:

  same data      axolotl's own prepared tensors (input_ids, labels, teacher
                 top-k token ids/logprobs/mask) - nothing re-tokenised
  same model     gemma-3-4b-it, bf16, eager attention, gradient checkpointing
  same adapter   LoRA r16, alpha 32, dropout 0.05 on q/k/v/o (PEFT keeps the
                 adapter weights in fp32, its default)
  same loss      axolotl's fused LigerFusedLinearKLTopKLogprobLoss, 0.5 KD +
                 0.5 CE, T=1, top-k renormalised, called as AxolotlKDTrainer
                 calls it, divided by the label tokens in the optimiser step
  same schedule  AdamW (fused), lr 2e-4, betas 0.9/0.999, eps 1e-8, no weight
                 decay, grad-norm clip 1.0, cosine with 10 warmup steps,
                 micro-batch 8 x 2 accumulation, 3 epochs

What cannot be identical: the order rows are drawn in (seeded, but not
axolotl's sampler) and the padding collator (right-padded here; axolotl's pads
the same way but is not reused).

Two deliberate departures, after the first all-bf16 run trained cleanly for 378
steps and then hit a NaN gradient at step 379:
  * the fused KD loss gets fp32 hidden states and an fp32 copy of the frozen
    output layer - what axolotl's working 4-bit path does, since
    prepare_model_for_kbit_training upcasts those layers. The model is bf16.
  * a step with a non-finite gradient is skipped (no optimiser update; the LR
    schedule keeps time) and logged, and the run aborts after MAX_SKIPS, so a
    persistently unstable run cannot pass as finished.

    python kd_lora_train.py <prepared_dir> <output_dir> [--model google/gemma-3-1b-it]

A smaller student than the teacher (e.g. Gemma 3 1B on the 4B teacher's data)
must index every teacher token id: the 4B vocabulary (262,208) is larger than
the 1B embedding (262,144). The largest id in the data is checked first; the
embedding is resized to the tokenizer, as axolotl does, if that covers it, and
the run stops if it does not.
"""

import argparse

import glob
import json

import pyarrow.compute as pc
import math
import sys
import time

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from axolotl.integrations.kd.kernels.liger import LigerFusedLinearKLTopKLogprobLoss
from kd_step0_check import lm_head_of

MICRO, ACCUM, EPOCHS, LR, WARMUP, CLIP, SEED = 8, 2, 3, 2e-4, 10, 1.0, 42
EVALS_PER_EPOCH = 2
MAX_SKIPS = 10


def collate(rows, pad_id, device):
    """Right-pad a micro-batch: pad token / -100 labels / masked-out teacher rows."""
    width = max(len(r["input_ids"]) for r in rows)
    k = len(rows[0]["target_token_ids"][0])
    ids, att, lab, tid, tlp, tmask = [], [], [], [], [], []
    for r in rows:
        n, pad = len(r["input_ids"]), width - len(r["input_ids"])
        ids.append(r["input_ids"] + [pad_id] * pad)
        att.append([1] * n + [0] * pad)
        lab.append(r["labels"] + [-100] * pad)
        tid.append(r["target_token_ids"] + [[0] * k] * pad)
        tlp.append(r["target_logprobs"] + [[0.0] * k] * pad)
        tmask.append(r["target_mask"] + [[0] * k] * pad)
    t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    return (t(ids, torch.long), t(att, torch.long), t(lab, torch.long),
            t(tid, torch.long), t(tlp, torch.float32), t(tmask, torch.long))


def kd_loss(model, head_fp32, fused, batch, n_items):
    ids, att, lab, tid, tlp, tmask = batch
    out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, return_dict=True, logits_to_keep=1)
    hidden = out.hidden_states[-1].float()
    return fused(head_fp32, hidden, tid, tlp, tmask, true_labels=lab) / n_items


def evaluate(model, head, fused, rows, pad_id):
    model.eval()
    total = tokens = 0
    with torch.no_grad():
        for i in range(0, len(rows), MICRO):
            batch = collate(rows[i:i + MICRO], pad_id, "cuda")
            n = int((batch[2] != -100).sum())
            total += kd_loss(model, head, fused, batch, 1).item(); tokens += n
    model.train()
    return total / tokens


def max_id(dataset, column):
    """Largest token id in a nested-list column, computed in Arrow."""
    col = dataset.data.column(column).combine_chunks()
    while hasattr(col, "flatten") and str(col.type).startswith("list"):
        col = col.flatten()
    return pc.max(col).as_py()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prepared_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    args = ap.parse_args()
    prepared, out_dir, MODEL = args.prepared_dir, args.output_dir, args.model
    sets = [d for d in (load_from_disk(x) for x in glob.glob(f"{prepared}/*/")) if "target_logprobs" in d.column_names]
    train, val = max(sets, key=len), min(sets, key=len)
    largest = max(max_id(d, c) for d in (train, val) for c in ("input_ids", "target_token_ids"))
    train_rows, val_rows = train.to_list(), val.to_list()
    print(f"train rows {len(train_rows)}, eval rows {len(val_rows)}", flush=True)

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda")
    base.config.use_cache = False
    rows = base.get_input_embeddings().num_embeddings
    if largest >= rows:
        if largest < len(tok):
            base.resize_token_embeddings(len(tok))
            print(f"largest token id {largest} >= {rows} embedding rows: resized to {len(tok)}", flush=True)
        else:
            sys.exit(f"largest token id {largest} is beyond the tokenizer ({len(tok)}) - cannot train {MODEL} on this data")
    else:
        print(f"largest token id {largest} < {rows} embedding rows: no resize", flush=True)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    model.train()
    # fp32 copy of the frozen output layer for the loss (see module docstring).
    head = lm_head_of(model).weight.detach().float()
    fused = LigerFusedLinearKLTopKLogprobLoss(0.5, 0.5, 1.0, 0.0, compute_ce_loss=True, normalize_topk=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
    micro_per_epoch = math.ceil(len(train_rows) / MICRO)
    steps_per_epoch = math.ceil(micro_per_epoch / ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, total_steps)
    eval_every = max(1, steps_per_epoch // EVALS_PER_EPOCH)
    print(f"optimiser steps {total_steps} ({steps_per_epoch}/epoch), eval every {eval_every}", flush=True)
    print(json.dumps({"step": 0, "eval_loss": round(evaluate(model, head, fused, val_rows, tok.pad_token_id), 4)}), flush=True)

    gen, step, skipped, t0 = torch.Generator().manual_seed(SEED), 0, 0, time.time()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(train_rows), generator=gen).tolist()
        micro_batches = [order[i:i + MICRO] for i in range(0, len(order), MICRO)]
        for s in range(0, len(micro_batches), ACCUM):
            group = [collate([train_rows[j] for j in mb], tok.pad_token_id, "cuda") for mb in micro_batches[s:s + ACCUM]]
            n_items = sum(int((b[2] != -100).sum()) for b in group)
            step_loss = 0.0
            for batch in group:
                loss = kd_loss(model, head, fused, batch, n_items)
                loss.backward()
                step_loss += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, CLIP)
            if torch.isfinite(grad_norm):
                opt.step()
            else:
                skipped += 1
                print(json.dumps({"step": step + 1, "skipped": "non-finite gradient", "skips_so_far": skipped}), flush=True)
                if skipped > MAX_SKIPS:
                    sys.exit(f"{skipped} non-finite steps - aborting, no adapter written")
            sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 10 == 0:
                print(json.dumps({"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": round(step_loss, 4),
                                  "grad_norm": round(grad_norm.item(), 3), "lr": sched.get_last_lr()[0]}), flush=True)
            if step % eval_every == 0 or step == total_steps:
                print(json.dumps({"step": step, "eval_loss": round(evaluate(model, head, fused, val_rows, tok.pad_token_id), 4)}), flush=True)

    model.save_pretrained(out_dir)
    print(json.dumps({"train_runtime_s": round(time.time() - t0), "skipped_steps": skipped, "peak_memory_gib":
                      round(torch.cuda.max_memory_allocated() / 2**30, 2), "saved": out_dir}), flush=True)


if __name__ == "__main__":
    main()
