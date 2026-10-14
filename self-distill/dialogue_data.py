"""Training data for the dialogue-task scaling curves.

    teacher  sample k constrained summaries per eligible train/validation
             dialogue from a teacher and keep, of those passing the checks, the
             one closest to the human reference (unigram F1) - sequence-level
             distillation with a verifier. Dialogues with no passing sample get
             no target and are left out of the constrained arms.
    build    write nested arms (prefixes of one shuffled order) for axolotl:
             ../datasets/dlg_<task>_curve/n<rows>/{train,validation} + ARMS.
             Reorder targets are the gold order; no teacher is involved.

    python dialogue_data.py teacher constrained google/gemma-3-27b-it ../datasets/samsum_raw ../results/dlg_teacher.jsonl --best-of 8
    python dialogue_data.py build reorder ../datasets/samsum_raw
    python dialogue_data.py build constrained ../datasets/samsum_raw --teacher ../results/dlg_teacher.jsonl

A teacher can be base + adapter (--adapter), merged on scratch as for
evaluation; e.g. the SAMSum full-data SFT model, trained on human summaries,
written as a separate variant:
    python dialogue_data.py teacher constrained google/gemma-3-4b-it ../datasets/samsum_raw ../results/dlg_teacher_sft.jsonl \
        --adapter ../models/gemma3-samsum-n6400-lora
    python dialogue_data.py build constrained ../datasets/samsum_raw --teacher ../results/dlg_teacher_sft.jsonl --name constrained_sft
"""

import argparse
import json
from pathlib import Path

import dialogue_tasks as dt

LADDER = [125, 250, 500, 1000, 2000, 4000, 8000]


def teacher(args):
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    parts = dt.splits_for(args.task, args.samsum)
    rows = [("train", r) for r in parts["train"]] + [("validation", r) for r in parts["validation"]]
    examples = [dt.BUILDERS[args.task](r) for _, r in rows]
    tok = AutoTokenizer.from_pretrained(args.model)
    mm = {} if "-1b-" in args.model else {"limit_mm_per_prompt": {"image": 0}}
    model_path = args.model
    if args.adapter:
        from dialogue_pilot import merged
        model_path = merged(args.model, args.adapter)
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88,
              enforce_eager=True, seed=42, **mm)
    texts = [tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False,
                                     add_generation_prompt=True) for ex in examples]
    outs = llm.generate(texts, SamplingParams(n=args.best_of, temperature=0.8, top_p=0.95,
                                              max_tokens=128, seed=42))
    kept, f1s = 0, []
    with open(args.out, "w") as f:
        for (split, _), ex, out in zip(rows, examples, outs):
            passing = [c.text.strip() for c in out.outputs if dt.check_constrained(ex, c.text.strip())["success"]]
            target = max(passing, key=lambda t: dt.unigram_f1(t, ex["reference"])) if passing else None
            kept += split == "train" and target is not None
            if target is not None:
                f1s.append(dt.unigram_f1(target, ex["reference"]))
            f.write(json.dumps({"id": ex["id"], "split": split, "target": target,
                                "n_pass": len(passing)}, ensure_ascii=False) + "\n")
    n_train = sum(s == "train" for s, _ in rows)
    teacher_name = args.model + (f" + {args.adapter}" if args.adapter else "")
    print(f"{teacher_name}: {kept}/{n_train} train dialogues have a verified target ({kept / n_train:.3f}); "
          f"targets' unigram F1 with the reference {sum(f1s) / max(len(f1s), 1):.3f}")


def rationales(args):
    """Rationales for the gold reorder answers from a teacher (best of k samples
    whose steps match the gold order)."""
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    parts = dt.splits_for("reorder", args.samsum)
    examples = [dt.reorder_example(r) for r in parts["train"]]
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88,
              enforce_eager=True, seed=42, limit_mm_per_prompt={"image": 0})
    texts = [tok.apply_chat_template([{"role": "user", "content": dt.reorder_rationale_request(ex)}],
                                     tokenize=False, add_generation_prompt=True) for ex in examples]
    outs = llm.generate(texts, SamplingParams(n=args.best_of, temperature=0.7, top_p=0.95, max_tokens=320, seed=42))
    kept = 0
    with open(args.out, "w") as f:
        for ex, out in zip(examples, outs):
            good = [c.text.strip() for c in out.outputs if dt.rationale_matches(c.text, ex["gold"])]
            kept += bool(good)
            f.write(json.dumps({"id": ex["id"], "rationale": min(good, key=len) if good else None}) + "\n")
    print(f"{args.model}: {kept}/{len(examples)} train dialogues have a rationale matching the gold order")


def build(args):
    from datasets import Dataset, DatasetDict, load_from_disk
    from transformers import AutoTokenizer
    parts = dt.splits_for(args.task, args.samsum)
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    targets = {}
    make = dt.BUILDERS[args.task]
    reasons = ({r["id"]: r["rationale"] for r in map(json.loads, open(args.rationales)) if r["rationale"]}
               if args.rationales else {})
    if args.task.startswith(("constrained", "xsumc")):
        targets = {r["id"]: r["target"] for r in map(json.loads, open(args.teacher)) if r["target"]}
        answer = lambda ex: targets.get(ex["id"])
    else:
        answer = lambda ex: json.dumps(ex["gold"])

    def rows(split, working=False):
        """One list per dialogue: the plain example, plus the working example
        when --rationale is set (train only; validation stays the test format)."""
        out = []
        for r in parts[split]:
            ex = make(r)
            if (target := answer(ex)) is not None:
                group = [{"messages": [{"role": "user", "content": ex["prompt"]},
                                       {"role": "assistant", "content": target}]}]
                if working and args.task == "reorder" and ex["id"] in reasons:
                    w = dt.reorder_working_example(ex, reasons[ex["id"]])
                    if len(tok(tok.apply_chat_template(w["messages"], tokenize=False))["input_ids"]) <= 1000:
                        group.append(w)
                elif working and args.task != "reorder":
                    group.append(dt.working_example(ex, target))
                out.append(group)
        return out

    train = rows("train", working=args.rationale or bool(args.rationales))
    val = [g[0] for g in rows("validation")]
    flat = lambda groups: [row for g in groups for row in g]
    longest = max(len(tok.apply_chat_template(r["messages"], tokenize=False)) for r in flat(train) + val)
    ladder = sorted((n for n in LADDER if n <= len(train)), reverse=True)
    root = Path(args.samsum).parent / f"dlg_{args.name or args.task}_curve"
    for n in ladder:
        DatasetDict({"train": Dataset.from_list(flat(train[:n])),
                     "validation": Dataset.from_list(val)}).save_to_disk(str(root / f"n{n}"))
    (root / "ARMS").write_text("".join(f"n{n}\n" for n in ladder))
    lens = [len(tok(tok.apply_chat_template(r["messages"], tokenize=False))["input_ids"]) for r in flat(train) + val]
    print(f"{args.task}: {len(train)} train dialogues ({len(flat(train))} rows), {len(val)} validation; arms {ladder}; "
          f"longest {max(lens)} tokens (chars {longest})")


def split(args):
    """Two disjoint halves of one curve arm, for data-parallel experts.

    half_a is the first n/2 rows of arm n<n>, i.e. exactly the nested arm
    n<n/2>; half_b is the rest. Same validation set as the curve."""
    from datasets import DatasetDict, load_from_disk
    root = Path(args.samsum).parent
    src = load_from_disk(str(root / f"dlg_{args.curve}_curve" / f"n{args.n}"))
    half = args.n // 2
    out = root / f"dlg_{args.curve}_split"
    for name, rows in (("half_a", range(half)), ("half_b", range(half, args.n))):
        DatasetDict({"train": src["train"].select(rows), "validation": src["validation"]}).save_to_disk(str(out / name))
    (out / "ARMS").write_text("half_a\nhalf_b\n")
    print(f"{out}: half_a = rows 0-{half - 1} (= n{half}), half_b = rows {half}-{args.n - 1}")


def kdprompts(args):
    """Prompts for gold-target KD on a curve arm (generate_teacher_logprobs.py
    --targets gold): the teacher and the student see the same prompt, and the
    gold answer is the `summary` the teacher scores."""
    from datasets import Dataset, DatasetDict, load_from_disk
    root = Path(args.samsum).parent
    arm = load_from_disk(str(root / f"dlg_{args.curve}_curve" / f"n{args.n}"))

    def rows(split):
        out = []
        for i, r in enumerate(arm[split]):
            user, gold = r["messages"][0], r["messages"][1]["content"]
            out.append({"id": f"{split}-{i}", "summary": gold, "teacher_messages": [user], "messages_short": [user]})
        return Dataset.from_list(out)

    dest = root / f"dlg_{args.curve}_kd_prompts"
    DatasetDict({"train": rows("train"), "validation": rows("validation")}).save_to_disk(str(dest))
    print(f"{dest}: {len(arm['train'])} train, {len(arm['validation'])} validation prompts with gold targets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher"); t.add_argument("task", choices=["constrained", "constrained2", "xsumc"])
    t.add_argument("model"); t.add_argument("samsum"); t.add_argument("out")
    t.add_argument("--best-of", type=int, default=8)
    t.add_argument("--adapter")
    b = sub.add_parser("build"); b.add_argument("task", choices=["reorder", "constrained", "constrained2", "xsumc"])
    b.add_argument("samsum"); b.add_argument("--teacher")
    b.add_argument("--name", help="arm directory dlg_<name>_curve; defaults to the task")
    kp = sub.add_parser("kdprompts"); kp.add_argument("curve"); kp.add_argument("samsum"); kp.add_argument("n", type=int)
    ra = sub.add_parser("rationales"); ra.add_argument("model"); ra.add_argument("samsum"); ra.add_argument("out")
    ra.add_argument("--best-of", type=int, default=4)
    sp = sub.add_parser("split"); sp.add_argument("curve"); sp.add_argument("samsum"); sp.add_argument("n", type=int)
    b.add_argument("--rationales", help="teacher rationales (dialogue_data.py rationales) for reorder step-by-step")
    b.add_argument("--rationale", action="store_true",
                   help="add a step-by-step working example per dialogue (constrained tasks)")
    args = ap.parse_args()
    {"teacher": teacher, "build": build, "split": split, "kdprompts": kdprompts, "rationales": rationales}[args.cmd](args)
