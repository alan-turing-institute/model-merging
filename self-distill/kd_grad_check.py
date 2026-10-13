"""Is the fused KD loss's gradient finite in bf16?

The bf16 self-distilled LoRA (self-distillation_xsum_lora_bf16.yaml) had a NaN
gradient at its very first step, before any update; the 4-bit run, identical
otherwise, trained cleanly. The visible difference: axolotl's 4-bit path runs
prepare_model_for_kbit_training, which upcasts the unquantised layers to fp32,
so the fused loss there works on fp32 hidden states and an fp32 output layer.
This runs that loss and its backward pass on real eval rows in both precisions.

    python kd_grad_check.py <prepared_dir> [--rows 8]
"""

import argparse
import glob

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

from axolotl.integrations.kd.kernels.liger import LigerFusedLinearKLTopKLogprobLoss
from kd_step0_check import lm_head_of


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("prepared_dir")
    p.add_argument("--rows", type=int, default=8)
    a = p.parse_args()
    ds = min((d for d in (load_from_disk(x) for x in glob.glob(f"{a.prepared_dir}/*/"))
              if "target_logprobs" in d.column_names), key=len)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", dtype=torch.bfloat16,
                                                 attn_implementation="eager", device_map="cuda").eval()
    head = lm_head_of(model)
    fused = LigerFusedLinearKLTopKLogprobLoss(0.5, 0.5, 1.0, 0.0, compute_ce_loss=True, normalize_topk=True)
    for dtype in (torch.bfloat16, torch.float32):
        bad_loss = bad_grad = 0
        for row in ds.select(range(a.rows)):
            ids = torch.tensor(row["input_ids"], device="cuda")[None]
            with torch.no_grad():
                H = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[-1]
            H = H.detach().to(dtype).requires_grad_(True)
            W = head.weight.detach().to(dtype).requires_grad_(True)
            loss = fused(W, H,
                         torch.tensor(row["target_token_ids"], device="cuda").long()[None],
                         torch.tensor(row["target_logprobs"], device="cuda").float()[None],
                         torch.tensor(row["target_mask"], device="cuda")[None],
                         true_labels=torch.tensor(row["labels"], device="cuda")[None])
            loss.backward()
            bad_loss += not torch.isfinite(loss).item()
            bad_grad += not (torch.isfinite(H.grad).all() and torch.isfinite(W.grad).all()).item()
        print(f"{str(dtype):15} rows {a.rows}: non-finite loss {bad_loss}, non-finite gradient {bad_grad}")


if __name__ == "__main__":
    main()
