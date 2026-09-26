"""Per-example adapter routing, against merging, on the cross-task pair.

THE QUESTION. Merging a crime classifier with a summariser produces one model
that does both. Routing keeps both experts and picks one per example. Routing
cannot exceed each expert on its own task - that is the oracle ceiling - and the
cross-task sweep already measured merging at 99.6% and 98.6% of exactly that.
So this quantifies a cost already known to be small; it does not look for a
large effect.

WHY IT IS STILL WORTH MEASURING. It converts a retention ratio into an absolute
cost, and the router is the component a sparse MoE would need. mergekit-moe
cannot emit a Gemma architecture, so this is how the routing question is asked
on the experts every other result in this project uses.

WHY NOT SAME-DIRECTION EXPERTS. The quarter shards are random draws from one
distribution, so no held-out example belongs to any particular shard and there
is nothing for a router to condition on. Routing needs an identifiable task.

THE ROUTER. Prompt-prefix matching. The two task prompts are entirely distinct,
so the router is perfect by construction and this measures the oracle ceiling
rather than a realistic router. That is deliberate: a weak router would confound
the ceiling with its own errors. Router accuracy is reported so the assumption
is visible rather than implied.

    uv run --project evaluate python experiments/adapter_routing.py \
      --crime-adapter ../models/gemma3-crime-full-lora \
      --xsum-adapter ../models/gemma3-xsum-full-lora \
      --crime-data ../datasets/crime_dataset/test \
      --xsum-data ../datasets/xsum_dataset/test --limit 200
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluate"))

CRIME_CUE = "Is the following Reddit post about crime?"
XSUM_CUE = "Summarise the following BBC News article"


def route(prompt):
    """Which expert handles this example. Returns 'crime', 'xsum' or None."""
    if prompt.startswith(CRIME_CUE):
        return "crime"
    if prompt.startswith(XSUM_CUE):
        return "xsum"
    return None


def generate(model, tokenizer, prompts, max_new_tokens, batch_size):
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True) for p in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        n = enc["input_ids"].shape[1]
        out += [tokenizer.decode(r, skip_special_tokens=True).strip() for r in gen[:, n:]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--crime-adapter", required=True)
    ap.add_argument("--xsum-adapter", required=True)
    ap.add_argument("--crime-data", required=True)
    ap.add_argument("--xsum-data", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from evaluate import build_messages, convert_to_label
    from rouge_score import rouge_scorer

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto")
    # Both adapters live in one model; set_adapter switches which is active, so
    # the base weights are held once rather than twice.
    model = PeftModel.from_pretrained(model, args.crime_adapter, adapter_name="crime")
    model.load_adapter(args.xsum_adapter, adapter_name="xsum")
    model.eval()

    crime = load_from_disk(args.crime_data).select(range(args.limit))
    xsum = load_from_disk(args.xsum_data).select(range(args.limit))

    # --- routing decisions, reported rather than assumed -------------------
    crime_prompts = [build_messages(t)[0]["content"] for t in crime["posts_name"]]
    xsum_prompts = list(xsum["prompt"])
    routed = [route(p) for p in crime_prompts] + [route(p) for p in xsum_prompts]
    truth = ["crime"] * len(crime_prompts) + ["xsum"] * len(xsum_prompts)
    correct = sum(r == t for r, t in zip(routed, truth))
    print(f"router accuracy {correct}/{len(truth)} = {correct/len(truth):.4f}")
    if correct != len(truth):
        print("  routing is not perfect; the ceiling below is not an oracle")

    # --- each expert on the task it was routed for -------------------------
    model.set_adapter("crime")
    preds = generate(model, tok, crime_prompts, 5, args.batch_size)
    labels = [convert_to_label(p) for p in preds]
    gold = list(crime["label"])
    acc = sum(l == g for l, g in zip(labels, gold) if l is not None) / len(gold)
    unparsed = sum(1 for l in labels if l is None)
    print(f"routed crime  accuracy {acc:.4f}  unparsed {unparsed}/{len(gold)}")

    model.set_adapter("xsum")
    preds = generate(model, tok, xsum_prompts, 64, args.batch_size)
    sc = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    refs = list(xsum["summary"])
    r1 = sum(sc.score(r, p)["rouge1"].fmeasure for p, r in zip(preds, refs)) / len(refs)
    words = sum(len(p.split()) for p in preds) / len(preds)
    print(f"routed xsum   ROUGE-1  {r1:.4f}  words {words:.1f}")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "router_accuracy": correct / len(truth),
            "routed_crime_accuracy": acc, "routed_crime_unparsed": unparsed,
            "routed_xsum_rouge1": r1, "routed_xsum_words": words,
            "n_per_task": args.limit,
        }, indent=2))
        print("saved", args.output)


if __name__ == "__main__":
    main()
