"""Sets a task MoE's router from example prompts: gate_mode prompt_mean.

mergekit-moe's gate_mode hidden builds each layer's router row for an expert
from the base model's hidden state at the LAST token of each positive prompt,
taken from the residual stream entering the layer, with the prompts as raw text;
hidden_avg averages over every position, padding included. The router at run
time sees something else: every token's state after attention and
post_attention_layernorm, in chat-templated inputs. This builds the router from
exactly that:

  for each expert (task), for each of its example prompts (make_moe_prompts.py:
  real training inputs in the evaluation's format, chat-templated, tokenised
  as Inspect does), run the base model on the prompt alone (no padding) and
  take, at every layer, the mean over its tokens of the router's input (the
  input of the layer's MLP); average over the prompts: mu[layer, expert].
  Router row = mu[layer, expert] - mean over experts of mu[layer, :].

Subtracting the mean over experts does not change which expert scores highest
(it lowers every expert's score for a token by the same amount), but it keeps
the scores small, so that bf16 does not round away the differences between
experts; unlike mergekit's L2 normalisation it adds no bias towards the expert
with the smallest vector. Each token then goes to the expert whose mean input
it is most aligned with. As with mergekit, the means come from the base model,
not the MoE, whose earlier layers' MLPs differ.

The MoE is built first by mergekit-moe (any gate_mode; random is used, since
its router is replaced); this overwrites model.layers.<i>.mlp.gate.weight in
its safetensors and writes router.json describing the router. moe.sbatch runs
it with --router-prompts.

Usage, from merge/:  uv run python set_moe_router.py <moe_dir> <prompts.json>
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def expert_tasks(moe_dir):
    cfg = yaml.safe_load(Path(moe_dir, "mergekit_moe_config.yml").read_text())
    tasks = [re.search(r"models/([a-z0-9]+)/seed_\d", e["source_model"]).group(1) for e in cfg["experts"]]
    return tasks, cfg["base_model"]


@torch.no_grad()
def router_inputs_mean(model, tokenizer, texts):
    """(layers, hidden): per layer, the router's input averaged over each
    text's tokens, then over the texts."""
    layers = model.model.layers
    acc = torch.zeros(len(layers), model.config.hidden_size, dtype=torch.float64)
    current = {}
    hooks = [layer.mlp.register_forward_pre_hook(
        lambda mod, args, i=i: current.__setitem__(i, args[0][0].double().mean(0).cpu()))
        for i, layer in enumerate(layers)]
    try:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)  # tokenizer defaults, as Inspect
            current.clear()
            model(input_ids=ids)
            acc += torch.stack([current[i] for i in range(len(layers))])
    finally:
        for h in hooks:
            h.remove()
    return acc / len(texts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("moe", help="the MoE directory (built by mergekit-moe)")
    ap.add_argument("prompts", help="moe/prompts_<model>.json (make_moe_prompts.py)")
    args = ap.parse_args()

    tasks, base = expert_tasks(args.moe)
    spec = json.loads(Path(args.prompts).read_text())
    if spec["base_model"] != base:
        raise SystemExit(f"{args.prompts} is for {spec['base_model']}, the MoE's base is {base}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda").eval()
    mu = torch.stack([router_inputs_mean(model, tokenizer, spec["prompts"][t]) for t in tasks], dim=1)  # (L, E, H)
    rows = mu - mu.mean(dim=1, keepdim=True)
    del model
    torch.cuda.empty_cache()

    # Overwrite each layer's router in the shard that holds it.
    index = json.loads(Path(args.moe, "model.safetensors.index.json").read_text())["weight_map"]
    by_shard = defaultdict(dict)
    for i in range(rows.shape[0]):
        key = f"model.layers.{i}.mlp.gate.weight"
        by_shard[index[key]][key] = rows[i]
    for shard, new in by_shard.items():
        path = Path(args.moe, shard)
        tensors = load_file(str(path))
        for key, value in new.items():
            old = tensors[key]
            if old.shape != value.shape:
                raise SystemExit(f"{key}: shape {tuple(old.shape)}, router computed as {tuple(value.shape)}")
            tensors[key] = value.to(old.dtype).contiguous()
        tmp = path.with_name(path.name + ".tmp")
        save_file(tensors, str(tmp), metadata={"format": "pt"})
        tmp.replace(path)
    # Read back: every router now holds the computed rows.
    for shard, new in by_shard.items():
        back = load_file(str(Path(args.moe, shard)))
        for key, value in new.items():
            assert torch.equal(back[key], value.to(back[key].dtype)), key

    cos = torch.nn.functional.cosine_similarity(mu[:, :, None, :], mu[:, None, :, :], dim=-1)  # (L, E, E)
    off = cos[:, ~torch.eye(len(tasks), dtype=torch.bool)]
    record = {
        "gate_mode": "prompt_mean",
        "script": "merge/set_moe_router.py",
        "rule": "row[layer, expert] = mu[layer, expert] - mean over experts of mu[layer, :], where mu is the mean "
                "over the expert's example prompts of the mean over their tokens of the layer's MLP input "
                "(the router's input), from the base model; prompts run one at a time, tokenised as Inspect does",
        "base_model": base,
        "experts": tasks,
        "prompts_file": str(Path(args.prompts).resolve()),
        "prompts_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "prompts_per_expert": {t: len(spec["prompts"][t]) for t in tasks},
        "xsum_prompt": spec.get("xsum_prompt"),
        "examples": spec.get("examples"),
        "mean_cosine_between_expert_means": {"min_layer": off.mean(1).min().item(), "max_layer": off.mean(1).max().item()},
    }
    Path(args.moe, "router.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"set {rows.shape[0]} routers for experts {tasks}; mean cosine between expert means per layer "
          f"{record['mean_cosine_between_expert_means']['min_layer']:.3f}-"
          f"{record['mean_cosine_between_expert_means']['max_layer']:.3f}")


if __name__ == "__main__":
    main()
