"""Distilling step-by-step on plain XSum (Hsieh et al., 2023, labelled setting).

    rationales <teacher> <out.jsonl>   the teacher, shown each training article
                                       and its reference summary, lists the 2-4
                                       facts from the article that the summary
                                       states (rationalisation)
    build <rationales.jsonl>           nested subsets of the 4,000 xsum_screen
                                       training articles (1,000/2,000/4,000): a
                                       supervised curve (article -> reference) and
                                       a step-by-step curve (the same rows plus a
                                       key-facts example each), as
                                       ../datasets/xsum_{sup,sbs}_curve

Testing keeps the plain short prompt (xsum_task.py, prompt_variant short).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = [1000, 2000, 4000]
TEACHER_ASK = ("Below is a BBC News article and its one-sentence summary. List the 2 to 4 facts from the "
               "article that the summary states, each as a short phrase in the article's own words, one per "
               "line starting with '- '. Reply with the list only.\n\nArticle:\n\n{doc}\n\nSummary: {summary}")
EXPLAIN = ("List the key facts from the BBC News article below that a one-sentence summary should state, "
           "one per line starting with '- ', instead of summarising.\n\n")


def full():
    from datasets import load_from_disk
    return load_from_disk(str(ROOT / "datasets" / "xsum_screen" / "full"))


def rationales(model, out):
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = full()["train"]
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88,
              enforce_eager=True, seed=42, limit_mm_per_prompt={"image": 0})
    prompts = [tok.apply_chat_template([{"role": "user", "content": TEACHER_ASK.format(doc=r["document"], summary=r["summary"])}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=160))
    kept = 0
    with open(out, "w") as f:
        for r, o in zip(rows, outs):
            lines = [l.strip() for l in o.outputs[0].text.strip().splitlines() if re.match(r"^\s*-\s+\S", l)]
            ok = 1 <= len(lines) <= 5 and o.outputs[0].finish_reason == "stop"
            kept += ok
            f.write(json.dumps({"id": r["id"], "rationale": "\n".join(lines) if ok else None}) + "\n")
    print(f"{model}: {kept}/{len(rows)} articles have a key-facts rationale")


def build(rationales_path):
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer
    data = full()
    why = {r["id"]: r["rationale"] for r in map(json.loads, open(rationales_path)) if r["rationale"]}
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    val = data["validation"].select_columns(["messages"])
    for name, with_why in (("sup", False), ("sbs", True)):
        root = ROOT / "datasets" / f"xsum_{name}_curve"
        for n in sorted(SIZES, reverse=True):
            rows = []
            for r in data["train"].select(range(n)):
                rows.append({"messages": r["messages"]})
                if with_why and r["id"] in why:
                    w = [{"role": "user", "content": EXPLAIN + r["document"]}, {"role": "assistant", "content": why[r["id"]]}]
                    if len(tok(tok.apply_chat_template(w, tokenize=False))["input_ids"]) <= 1000:
                        rows.append({"messages": w})
            DatasetDict({"train": Dataset.from_list(rows), "validation": val}).save_to_disk(str(root / f"n{n}"))
            print(f"{root.name}/n{n}: {len(rows)} rows")
        (root / "ARMS").write_text("".join(f"n{n}\n" for n in sorted(SIZES, reverse=True)))


if __name__ == "__main__":
    {"rationales": lambda: rationales(sys.argv[2], sys.argv[3]), "build": lambda: build(sys.argv[2])}[sys.argv[1]]()
