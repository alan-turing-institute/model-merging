"""Measure what merging actually costs at inference, against the alternatives.

The feasibility report claims a cross-task merge keeps both skills "at no extra
inference cost". That is an architectural argument - merging folds both task
vectors into the base weights, so the result is one dense model of the base
architecture - and until now it was only an argument. This measures it.

Four configurations, same hardware, same prompts, same decode settings:

    base              the untouched base model, the reference
    merged            the cross-task merge: one model, both skills
    base+1 adapter    a single expert served as base + LoRA, unmerged
    base+2 adapters   the routing alternative: both adapters resident,
                      switched per request, which is what an oracle router needs

The comparison that matters is `merged` against `base+2 adapters`, because those
are the two ways to serve both skills from one deployment. `base` bounds what is
achievable and `base+1` separates the cost of carrying an adapter at all from the
cost of carrying two.

Prompt length is reported separately for short (classification) and long
(summarisation) inputs, because LoRA overhead is per-token and the two tasks sit
at opposite ends of the range.
"""

import argparse
import json
import time

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 64
WARMUP_TOKENS = 8


def load_base(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")


def bench(label, build, tokenizer, prompt_sets, max_new_tokens=MAX_NEW_TOKENS):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build()
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    result = {"label": label, "params": params, "params_billions": round(params / 1e9, 4)}

    for set_name, prompts in prompt_sets.items():
        # One untimed pass: the first generate() pays for kernel autotuning and
        # cache allocation, which would otherwise be charged to whichever
        # configuration happened to run first.
        enc = tokenizer(prompts[0], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=WARMUP_TOKENS, do_sample=False)
        torch.cuda.synchronize()

        generated = 0
        start = time.perf_counter()
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            generated += out.shape[1] - enc["input_ids"].shape[1]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        result[set_name] = {
            "prompts": len(prompts),
            "new_tokens": generated,
            "seconds": round(elapsed, 2),
            "tokens_per_s": round(generated / elapsed, 2),
            "ms_per_prompt": round(1000 * elapsed / len(prompts), 1),
        }

    result["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--merged", required=True)
    ap.add_argument("--crime-adapter", required=True)
    ap.add_argument("--xsum-adapter", required=True)
    ap.add_argument("--crime-dataset", required=True)
    ap.add_argument("--xsum-dataset", required=True)
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3,
                    help="rounds per configuration, interleaved; reported as median and spread")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    crime = load_from_disk(args.crime_dataset).select(range(args.n_prompts))
    xsum = load_from_disk(args.xsum_dataset).select(range(args.n_prompts))
    # Take the rendered user turn so prompt lengths match what the evaluators
    # actually send, rather than a synthetic string.
    def user_text(row):
        msgs = row.get("messages") or row.get("messages_short") or []
        for m in msgs:
            if m.get("role") == "user":
                return m["content"]
        return row.get("document") or row.get("posts_name") or ""
    prompt_sets = {
        "short_prompts_crime": [user_text(r) for r in crime],
        "long_prompts_xsum": [user_text(r) for r in xsum],
    }
    for name, ps in prompt_sets.items():
        lens = [len(tokenizer(p)["input_ids"]) for p in ps]
        print(f"{name}: {len(ps)} prompts, {min(lens)}-{max(lens)} tokens "
              f"(median {sorted(lens)[len(lens)//2]})")

    def with_one_adapter():
        return PeftModel.from_pretrained(load_base(args.base), args.crime_adapter)

    def with_two_adapters():
        m = PeftModel.from_pretrained(load_base(args.base), args.crime_adapter,
                                      adapter_name="crime")
        m.load_adapter(args.xsum_adapter, adapter_name="xsum")
        m.set_adapter("crime")
        return m

    configs = [
        ("base", lambda: load_base(args.base)),
        ("merged_crosstask", lambda: load_base(args.merged)),
        ("base+1_adapter", with_one_adapter),
        ("base+2_adapters_routed", with_two_adapters),
    ]

    # ROUND-ROBIN, NOT ONE BLOCK PER CONFIGURATION. The first version ran each
    # configuration once in sequence and reported the base model at 9.2 tok/s
    # against the merge's 22.2 - a 2.4x gap between two models with the SAME
    # architecture and parameter count, which is impossible. Whatever ran first
    # was paying for the GPU's clock ramp. Interleaving the repeats spreads that
    # cost across every configuration instead of charging it to one, and the
    # spread across rounds shows whether it has actually gone.
    rounds = []
    for r in range(args.repeats):
        for label, build in configs:
            print(f"=== round {r+1}/{args.repeats}: {label} ===", flush=True)
            rounds.append(bench(label, build, tokenizer, prompt_sets))

    results = []
    for label, _ in configs:
        runs = [r for r in rounds if r["label"] == label]
        merged = {"label": label, "params": runs[0]["params"],
                  "params_billions": runs[0]["params_billions"],
                  "peak_gib": max(r["peak_gib"] for r in runs)}
        for set_name in prompt_sets:
            rates = sorted(r[set_name]["tokens_per_s"] for r in runs)
            merged[set_name] = {
                "tokens_per_s_median": rates[len(rates) // 2],
                "tokens_per_s_runs": rates,
                # If this stays wide, the measurement is still dominated by
                # something other than the model.
                "spread": round(rates[-1] - rates[0], 2),
            }
        print(json.dumps(merged, indent=2), flush=True)
        results.append(merged)

    by = {r["label"]: r for r in results}
    summary = {
        "configurations": results,
        "merged_vs_base_params": by["merged_crosstask"]["params"] - by["base"]["params"],
        "routed_vs_base_params": by["base+2_adapters_routed"]["params"] - by["base"]["params"],
    }
    for s in ("short_prompts_crime", "long_prompts_xsum"):
        summary[f"{s}_tokens_per_s"] = {r["label"]: r[s]["tokens_per_s_median"] for r in results}
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
