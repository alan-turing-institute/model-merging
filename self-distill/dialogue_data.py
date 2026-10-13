"""Training data for the dialogue-task scaling curves.

    teacher  sample k constrained summaries per eligible train/validation
             dialogue from a teacher and keep, of those passing the checks, the
             one closest to the human reference (unigram F1) - sequence-level
             distillation with a verifier. Dialogues with no passing sample get
             no target and are left out of the constrained arms.
    build    write nested arms (prefixes of one shuffled order) for axolotl:
             ../datasets/dlg_<task>_curve/n<rows>/{train,validation} + ARMS.
             Reorder targets are the gold order; no teacher is involved.

    python dialogue_data.py teacher google/gemma-3-27b-it ../datasets/samsum_raw ../results/dlg_teacher.jsonl --best-of 8
    python dialogue_data.py build reorder ../datasets/samsum_raw
    python dialogue_data.py build constrained ../datasets/samsum_raw --teacher ../results/dlg_teacher.jsonl
"""

import argparse
import json
from pathlib import Path

import dialogue_tasks as dt

LADDER = [125, 250, 500, 1000, 2000, 4000]


def teacher(args):
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    parts = dt.splits(load_from_disk(args.samsum))
    rows = [("train", r) for r in parts["train"]] + [("validation", r) for r in parts["validation"]]
    examples = [dt.constrained_example(r) for _, r in rows]
    tok = AutoTokenizer.from_pretrained(args.model)
    mm = {} if "-1b-" in args.model else {"limit_mm_per_prompt": {"image": 0}}
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88,
              enforce_eager=True, seed=42, **mm)
    texts = [tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False,
                                     add_generation_prompt=True) for ex in examples]
    outs = llm.generate(texts, SamplingParams(n=args.best_of, temperature=0.8, top_p=0.95,
                                              max_tokens=128, seed=42))
    kept = 0
    with open(args.out, "w") as f:
        for (split, _), ex, out in zip(rows, examples, outs):
            passing = [c.text.strip() for c in out.outputs if dt.check_constrained(ex, c.text.strip())["success"]]
            target = max(passing, key=lambda t: dt.unigram_f1(t, ex["reference"])) if passing else None
            kept += split == "train" and target is not None
            f.write(json.dumps({"id": ex["id"], "split": split, "target": target,
                                "n_pass": len(passing)}, ensure_ascii=False) + "\n")
    n_train = sum(s == "train" for s, _ in rows)
    print(f"{args.model}: {kept}/{n_train} train dialogues have a verified target ({kept / n_train:.3f})")


def build(args):
    from datasets import Dataset, DatasetDict, load_from_disk
    from transformers import AutoTokenizer
    parts = dt.splits(load_from_disk(args.samsum))
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    targets = {}
    if args.task == "constrained":
        targets = {r["id"]: r["target"] for r in map(json.loads, open(args.teacher)) if r["target"]}
        make, answer = dt.constrained_example, lambda ex: targets.get(ex["id"])
    else:
        make, answer = dt.reorder_example, lambda ex: json.dumps(ex["gold"])

    def rows(split):
        out = []
        for r in parts[split]:
            ex = make(r)
            if (target := answer(ex)) is not None:
                out.append({"messages": [{"role": "user", "content": ex["prompt"]},
                                         {"role": "assistant", "content": target}]})
        return out

    train, val = rows("train"), rows("validation")
    longest = max(len(tok.apply_chat_template(r["messages"], tokenize=False)) for r in train + val)
    ladder = sorted((n for n in LADDER if n <= len(train)), reverse=True)
    root = Path(args.samsum).parent / f"dlg_{args.task}_curve"
    for n in ladder:
        DatasetDict({"train": Dataset.from_list(train[:n]),
                     "validation": Dataset.from_list(val)}).save_to_disk(str(root / f"n{n}"))
    (root / "ARMS").write_text("".join(f"n{n}\n" for n in ladder))
    lens = [len(tok(tok.apply_chat_template(r["messages"], tokenize=False))["input_ids"]) for r in train + val]
    print(f"{args.task}: {len(train)} train rows, {len(val)} validation; arms {ladder}; "
          f"longest {max(lens)} tokens (chars {longest})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher"); t.add_argument("model"); t.add_argument("samsum"); t.add_argument("out")
    t.add_argument("--best-of", type=int, default=8)
    b = sub.add_parser("build"); b.add_argument("task", choices=["reorder", "constrained"])
    b.add_argument("samsum"); b.add_argument("--teacher")
    args = ap.parse_args()
    {"teacher": teacher, "build": build}[args.cmd](args)
