"""Which parameters get a non-finite gradient in the unquantised KD LoRA step?

The unquantised self-distilled LoRA has a NaN gradient at its first step in
bf16 and in fp32, with or without padding or gradient checkpointing, while the
4-bit LoRA and main's bf16 full fine-tune train cleanly; the fused KD loss on
its own has finite gradients. This rebuilds the training step outside axolotl -
the same model, LoRA (r16/a32 on q,k,v,o) and fused KD loss, called the way
AxolotlKDTrainer.compute_loss calls it - on one real training row, and lists
every parameter whose gradient is not finite, for:

  bf16 base + KD loss      (the failing configuration)
  4-bit base + KD loss     (the one that trains)
  bf16 base + ordinary LM loss on the model's own logits

    python kd_lora_nan_check.py <prepared_dir>
"""

import glob
import sys

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from axolotl.integrations.kd.kernels.liger import LigerFusedLinearKLTopKLogprobLoss
from kd_step0_check import lm_head_of


def build(quantised):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) if quantised else None
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", dtype=torch.bfloat16,
                                                 attn_implementation="eager", quantization_config=quant,
                                                 device_map="cuda")
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    peft = get_peft_model(model, lora)
    peft.train()
    return peft


def step(peft, row, loss_kind):
    dev = "cuda"
    ids = torch.tensor(row["input_ids"], device=dev)[None]
    labels = torch.tensor(row["labels"], device=dev)[None]
    n_items = int((labels != -100).sum())
    with torch.autograd.detect_anomaly(check_nan=True):
        try:
            if loss_kind == "kd":
                out = peft(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True,
                           return_dict=True, logits_to_keep=1)
                head = lm_head_of(peft)
                H = out.hidden_states[-1].to(head.weight.dtype)
                fused = LigerFusedLinearKLTopKLogprobLoss(0.5, 0.5, 1.0, 0.0, compute_ce_loss=True,
                                                          normalize_topk=True)
                loss = fused(head.weight, H,
                             torch.tensor(row["target_token_ids"], device=dev).long()[None],
                             torch.tensor(row["target_logprobs"], device=dev).float()[None],
                             torch.tensor(row["target_mask"], device=dev)[None],
                             true_labels=labels) / n_items
            else:
                loss = peft(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            loss.backward()
            anomaly = None
        except RuntimeError as e:
            anomaly = str(e).splitlines()[0][:220]
    bad = [n for n, p in peft.named_parameters()
           if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
    none = sum(1 for n, p in peft.named_parameters() if p.requires_grad and p.grad is None)
    trainable = sum(1 for _, p in peft.named_parameters() if p.requires_grad)
    return loss.item() if anomaly is None else float("nan"), anomaly, bad, none, trainable


def main():
    prepared = sys.argv[1]
    ds = max((d for d in (load_from_disk(x) for x in glob.glob(f"{prepared}/*/"))
              if "target_logprobs" in d.column_names), key=len)
    row = ds[0]
    for label, quantised, kind in (("bf16 base + KD loss", False, "kd"),
                                   ("4-bit base + KD loss", True, "kd"),
                                   ("bf16 base + ordinary LM loss", False, "lm")):
        peft = build(quantised)
        loss, anomaly, bad, none, trainable = step(peft, row, kind)
        print(f"\n=== {label} ===")
        print(f"loss {loss:.4f} | trainable tensors {trainable} | no gradient {none} | non-finite gradient {len(bad)}")
        if anomaly:
            print("anomaly detection:", anomaly)
        for n in bad[:8]:
            print("  non-finite:", n)
        del peft
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
