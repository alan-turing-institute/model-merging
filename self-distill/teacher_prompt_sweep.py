"""Teacher-prompt comparison for XSum self-distillation: generation.

The self-distilled student reaches its teacher's ROUGE against the references
(0.241 vs 0.235), so raising it means changing the teacher's targets, not the
student's training. This generates teacher summaries for the 250 validation
articles (xsum_screen, cut to 512 words, never used for testing) under several
prompts, with main's teacher settings - gemma-3-4b-it, greedy, 128 tokens - so
evaluate/teacher_prompt_score.py can compare them before anything is retrained.

  P0_current        main's TEACHER_SYSTEM + TEACHER_ASK, unchanged
  P1_bbc_style      describes how XSum summaries are written (past / present
                    perfect, names, places, numbers, ~20 words)
  P2_fewshot        three XSum training article/summary pairs as demonstrations
  P3_fewshot_faithful   P2 plus "use only facts stated in the article"
  P4_student_short  the student's short prompt, no rules (baseline)

    python teacher_prompt_sweep.py <xsum_screen/full> <output_dir>
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

spec = importlib.util.spec_from_file_location("prep", Path(__file__).with_name("prepare_data_XSum.py"))
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
_kd = importlib.util.spec_from_file_location("kdp", Path(__file__).with_name("prepare_kd_screen_prompts.py"))
kdp = importlib.util.module_from_spec(_kd); _kd.loader.exec_module(kdp)

MODEL = "google/gemma-3-4b-it"

FEWSHOT = ("You write the one-sentence summary that appears above a BBC News article. "
           "Reply with the summary sentence only.")
FAITHFUL = " Use only facts stated in the article."


def demonstrations(train, n=3):
    """The first n training articles of 150-300 words, with their reference summaries."""
    picked = []
    for row in train:
        if 150 <= len(row["document"].split()) <= 300:
            picked.append(row)
        if len(picked) == n:
            return picked
    raise SystemExit("not enough short training articles for demonstrations")


def prompts(document, demos):
    ask = f"Article:\n\n{document}\n\nSummarise the article above in one sentence."
    shots = []
    for d in demos:
        shots += [{"role": "user", "content": f"Article:\n\n{d['document']}\n\nSummarise the article above in one sentence."},
                  {"role": "assistant", "content": d["summary"]}]
    return {
        "P0_current": prep.convert_to_prompt({"document": document})["teacher_messages"],
        "P1_bbc_style": kdp.bbc_teacher_messages(document),
        "P2_fewshot": [{"role": "system", "content": FEWSHOT}] + shots + [{"role": "user", "content": ask}],
        "P3_fewshot_faithful": [{"role": "system", "content": FEWSHOT + FAITHFUL}] + shots + [{"role": "user", "content": ask}],
        "P4_student_short": prep.convert_to_prompt({"document": document})["messages_short"],
    }


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    splits = load_from_disk(str(source))
    val, demos = splits["validation"], demonstrations(splits["train"])
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
              enforce_eager=True, seed=42)
    params = SamplingParams(temperature=0.0, max_tokens=128, seed=42)
    per_row = [prompts(r["document"], demos) for r in val]
    for name in per_row[0]:
        texts = [tok.apply_chat_template(p[name], tokenize=False, add_generation_prompt=True) for p in per_row]
        outputs = llm.generate(texts, params)
        rows = [{"id": r["id"], "document": r["document"], "reference": r["summary"],
                 "prediction": o.outputs[0].text.strip(), "finish": o.outputs[0].finish_reason}
                for r, o in zip(val, outputs)]
        (out / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"{name}: {len(rows)} generated, {sum(x['finish'] == 'length' for x in rows)} hit the 128-token cap", flush=True)


if __name__ == "__main__":
    main()
