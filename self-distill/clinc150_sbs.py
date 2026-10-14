"""Distilling step-by-step on CLINC150 for Gemma 3 1B at reduced data.

    rationales <teacher> <out.jsonl>   the teacher, shown each query and its gold
                                       intent, writes one sentence on which words
                                       show that intent (rationalisation)
    build <rationales.jsonl>           nested, intent-balanced subsets of the
                                       15,000 training queries (500-4,000): a
                                       supervised curve and a step-by-step curve
                                       (the same rows plus one rationale example
                                       each), as ../datasets/clinc150_{sup,sbs}_curve

Subsets are round-robin across the 150 intents, so each prefix is balanced.
Testing keeps the plain prompt (clinc150_task.py, prompt_variant short).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = [500, 1000, 2000, 4000]
TEACHER_ASK = ('A user asked an assistant: "{q}"\nIts intent label is "{label}", one of 150 intent labels '
               "for an assistant covering banking, credit cards, travel, home, work, cars, food and small "
               "talk. In one sentence of at most 20 words, explain which words in the query show this "
               "intent. Reply with the sentence only.")
EXPLAIN = ("Explain in one sentence which words in the query below show its intent label, instead of "
           "answering.\n\n")


def ordered():
    """Training rows in round-robin intent order (deterministic)."""
    from datasets import load_from_disk
    full = load_from_disk(str(ROOT / "datasets" / "clinc150" / "full"))
    by_label = {}
    for i, r in enumerate(full["train"]):
        by_label.setdefault(r["messages"][1]["content"], []).append(i)
    labels = sorted(by_label)
    order = [by_label[l][k] for k in range(max(map(len, by_label.values()))) for l in labels if k < len(by_label[l])]
    return full, order


def rationales(model, out):
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    full, order = ordered()
    rows = [full["train"][i] for i in order[: max(SIZES)]]
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.88,
              enforce_eager=True, seed=42, limit_mm_per_prompt={"image": 0})
    prompts = [tok.apply_chat_template([{"role": "user", "content": TEACHER_ASK.format(
        q=r["text"], label=r["messages"][1]["content"])}], tokenize=False, add_generation_prompt=True) for r in rows]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=64))
    with open(out, "w") as f:
        for i, r, o in zip(order, rows, outs):
            text = o.outputs[0].text.strip()
            f.write(json.dumps({"row": i, "rationale": text if 0 < len(text.split()) <= 40 else None}) + "\n")
    print(f"{model}: {sum(1 for o in outs if o.outputs[0].text.strip())} rationales for {len(rows)} queries")


def build(rationales_path):
    from datasets import Dataset, DatasetDict
    full, order = ordered()
    why = {r["row"]: r["rationale"] for r in map(json.loads, open(rationales_path)) if r["rationale"]}
    val = full["validation"].select_columns(["messages"])
    for name, with_why in (("sup", False), ("sbs", True)):
        root = ROOT / "datasets" / f"clinc150_{name}_curve"
        for n in sorted(SIZES, reverse=True):
            rows = []
            for i in order[:n]:
                r = full["train"][i]
                rows.append({"messages": r["messages"]})
                if with_why and i in why:
                    query = r["messages"][0]["content"].split("Query: ", 1)[1]
                    rows.append({"messages": [{"role": "user", "content": EXPLAIN + f"Query: {query}"},
                                              {"role": "assistant", "content": why[i]}]})
            DatasetDict({"train": Dataset.from_list(rows), "validation": val}).save_to_disk(str(root / f"n{n}"))
        (root / "ARMS").write_text("".join(f"n{n}\n" for n in sorted(SIZES, reverse=True)))
        print(f"{root}: arms {sorted(SIZES, reverse=True)}; n{n} (the last built) has {len(rows)} rows")


if __name__ == "__main__":
    {"rationales": lambda: rationales(sys.argv[2], sys.argv[3]), "build": lambda: build(sys.argv[2])}[sys.argv[1]]()
