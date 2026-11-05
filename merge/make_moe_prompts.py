"""Example prompts for the prompt_mean MoE routers (set_moe_router.py).

For each task, N random training examples written exactly as the evaluation
writes its test prompts, as the user turn, rendered with the model's chat
template and the generation prompt (as Inspect's hf provider renders them), so
that the router is built from the same kind of input it will route:
  crime, wine, CLINC150  the training set's `messages` user turn, which is the
                         evaluation's prompt format (checked equal on the test
                         sets, 5 Nov)
  XSum                   --xsum-prompt eval (default): convert_to_prompt from
                         self-distill/prepare_data_XSum.py, as evaluate/
                         xsum_task.py builds its prompts ("about 18-24 words");
                         training: the dataset's stored messages_long, the
                         prompt the XSum experts were trained on ("20-25 words").
Writes moe/prompts_<model>.json: {"prompts": {task: [rendered text, ...]}, ...}.

Usage, from merge/ (needs datasets and transformers: evaluate/'s environment):
  uv run --project ../evaluate python make_moe_prompts.py <mistral|llama> [--n 8]
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
BASES = {"mistral": "mistralai/Mistral-7B-Instruct-v0.3", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
DATASETS = {
    "crime": REPO / "datasets/crime/crime_dataset",
    "wine": REPO / "datasets/wine/wine_dataset",
    "clinc150": REPO / "datasets/clinc150/clinc150_labels_dataset",
    "xsum": REPO / "datasets/xsum/xsum_prompts",
}


def xsum_convert_to_prompt():
    spec = importlib.util.spec_from_file_location("prepare_data_XSum", REPO / "self-distill/prepare_data_XSum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_to_prompt


def user_turn(task, row, xsum_prompt, convert):
    if task == "xsum":
        msgs = convert(row)["messages_long"] if xsum_prompt == "eval" else row["messages_long"]
    else:
        msgs = row["messages"]
    msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
    return [m for m in msgs if m["role"] == "user"][:1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=BASES)
    ap.add_argument("--n", type=int, default=8, help="examples per task")
    ap.add_argument("--seed", type=int, default=0, help="which training examples")
    ap.add_argument("--xsum-prompt", choices=["eval", "training"], default="eval")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASES[args.model])
    convert = xsum_convert_to_prompt()
    out = {"base_model": BASES[args.model], "n": args.n, "seed": args.seed, "xsum_prompt": args.xsum_prompt,
           "rendering": "chat template with add_generation_prompt, as text (BOS included)",
           "examples": {}, "prompts": {}}
    for task, path in DATASETS.items():
        train = load_from_disk(str(path))["train"]
        idx = sorted(random.Random(args.seed).sample(range(len(train)), args.n))
        out["examples"][task] = {"dataset": str(path.relative_to(REPO)) + ":train", "rows": idx}
        out["prompts"][task] = [
            tokenizer.apply_chat_template(user_turn(task, train[i], args.xsum_prompt, convert),
                                          add_generation_prompt=True, tokenize=False)
            for i in idx]
    path = Path(__file__).resolve().parent / "moe" / f"prompts_{args.model}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {args.n} prompts per task")


if __name__ == "__main__":
    main()
