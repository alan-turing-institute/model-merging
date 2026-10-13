"""Pilot for the harder dialogue tasks: how do untrained Gemma 3 models do?

Generates, for one model, greedy answers to both tasks in dialogue_tasks.py on
SAMSum validation dialogues (6-14 turns, three topic words available). With
--best-of k it also samples k constrained summaries per dialogue and records
whether any passes the checks - the yield of verified teacher targets.

    python dialogue_pilot.py google/gemma-3-27b-it ../datasets/samsum_raw <out_dir> --best-of 8
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import dialogue_tasks as dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("samsum")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--best-of", type=int, default=0)
    args = ap.parse_args()

    rows = [r for r in load_from_disk(args.samsum)["validation"] if dt.eligible(r)][: args.n]
    examples = [dt.constrained_example(r) for r in rows] + [dt.reorder_example(r) for r in rows]
    tok = AutoTokenizer.from_pretrained(args.model)
    text = lambda ex: tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                              tokenize=False, add_generation_prompt=True)
    # Gemma 3 4B and up are image-text models; text-only here, so no image slots.
    # The 1B is text-only and takes no multimodal settings.
    mm = {} if "-1b-" in args.model else {"limit_mm_per_prompt": {"image": 0}}
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True, seed=42, **mm)
    greedy = llm.generate([text(ex) for ex in examples], SamplingParams(temperature=0.0, max_tokens=128))
    for ex, out in zip(examples, greedy):
        ex["prediction"] = out.outputs[0].text.strip()
        ex["check"] = dt.CHECKS[ex["task"]](ex, ex["prediction"])

    if args.best_of:
        constrained = [ex for ex in examples if ex["task"] == "constrained"]
        sampled = llm.generate([text(ex) for ex in constrained],
                               SamplingParams(n=args.best_of, temperature=0.8, top_p=0.95,
                                              max_tokens=128, seed=42))
        for ex, out in zip(constrained, sampled):
            passing = [c.text.strip() for c in out.outputs
                       if dt.check_constrained(ex, c.text.strip())["success"]]
            ex.update(best_of=args.best_of, best_ok=bool(passing), passing=passing)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.model.split('/')[-1]}.jsonl"
    path.write_text("".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in examples))
    print(f"wrote {len(examples)} rows ({len(rows)} dialogues) to {path}")


if __name__ == "__main__":
    main()
