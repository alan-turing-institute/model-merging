"""Measure what the MoE's untrained router actually does.

mergekit-moe builds the gate from hidden states over a handful of prompts and
never trains it - its own CLI has a flag called
--i-understand-this-is-not-useful-without-training. So before investing in
router training, establish whether the router is the defect at all.

The question is simple: on medical text, how often does the router pick the
medical expert? Three outcomes, three different follow-ups.

  ~50/50 everywhere        the gate is noise; training it is the whole job
  strongly domain-split    the gate works; MoE quality is limited elsewhere
  collapsed to one expert  the MoE is effectively one model with dead weight,
                           and the eval will look like that expert alone

Expert 0 is Llama-2-7B-Chat, expert 1 is Meditron-7B, matching the order in
configs/moe-2way.yml.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Drawn from the six benchmarks the merges are scored on: three medical
# (medqa, medmcqa, pubmedqa) and three general (arc_challenge, hellaswag, mmlu).
MEDICAL = [
    "A 54-year-old man presents with crushing substernal chest pain radiating to the left arm.",
    "Which of the following antibiotics is first-line for community-acquired pneumonia?",
    "The patient's arterial blood gas shows pH 7.28, PaCO2 60 mmHg, HCO3 26 mEq/L.",
    "Metformin lowers hepatic glucose production primarily by activating AMP-kinase.",
    "A 3-year-old presents with barking cough and inspiratory stridor after a viral prodrome.",
    "Does the abstract's evidence support a causal link between the exposure and outcome?",
]
GENERAL = [
    "Which of the following is the primary reason that seasons occur on Earth?",
    "She opened the door, stepped outside, and immediately",
    "The capital of France is",
    "In economics, opportunity cost refers to",
    "A student pushes a box across a floor at constant velocity. The net force is",
    "Summarise the main argument of the passage in one sentence.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="cuda"
    )
    model.eval()

    cfg = model.config
    n_experts = getattr(cfg, "num_local_experts", None)
    print(f"model_type={cfg.model_type} experts={n_experts} "
          f"top_k={getattr(cfg, 'num_experts_per_tok', None)}")

    results = {}
    for label, prompts in (("medical", MEDICAL), ("general", GENERAL)):
        # per-layer count of tokens whose top-1 router choice is each expert
        counts = None
        for text in prompts:
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**ids, output_router_logits=True)
            # tuple of [num_tokens, num_experts], one per layer
            layers = out.router_logits
            if counts is None:
                counts = torch.zeros(len(layers), n_experts)
            for i, logits in enumerate(layers):
                pick = logits.float().argmax(dim=-1)
                counts[i] += torch.bincount(pick.cpu(), minlength=n_experts).float()
        frac = (counts / counts.sum(dim=1, keepdim=True)).tolist()
        results[label] = frac
        overall = counts.sum(0) / counts.sum()
        print(f"\n{label}: overall expert share "
              f"{[round(x, 3) for x in overall.tolist()]}")
        for i in (0, len(frac) // 2, len(frac) - 1):
            print(f"  layer {i:>2}: {[round(x, 3) for x in frac[i]]}")

    # The number that matters: does the medical expert get more medical tokens
    # than it gets general ones? If not, the gate carries no domain signal.
    med_on_med = sum(l[1] for l in results["medical"]) / len(results["medical"])
    med_on_gen = sum(l[1] for l in results["general"]) / len(results["general"])
    print(f"\nExpert 1 (Meditron) share: {med_on_med:.3f} on medical, "
          f"{med_on_gen:.3f} on general, separation {med_on_med - med_on_gen:+.3f}")
    results["separation"] = med_on_med - med_on_gen

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
