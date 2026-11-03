"""Write a mergekit-moe config whose routing prompts are the inputs themselves.

moe_mistral_mlp_2way.yml routes on three instructions per expert. In 85's run
that sent 25% of XSum prompt tokens to the XSum expert and 86% of crime tokens
to the crime expert: most tokens of an input are its article or post, not an
instruction, and the router was never shown any.

gate_mode hidden takes, per layer, the base model's hidden state at each
prompt's last token. In a causal model the hidden state at a position depends
only on the tokens before it, so the last token of an input cut short is the
state that token has inside the whole input. Cutting real inputs at random
points therefore samples the token states the router meets at evaluation:

  - inputs are training rows (not the test rows 85 scores), rendered as the
    evaluators render them: crime through evaluate/evaluate.py's
    build_messages, XSum from the `prompt` column, each as one user turn through
    the chat template with the generation prompt;
  - each is cut after a word chosen uniformly from its words, instruction
    included, or kept whole (ending in [/INST]), since every prompt token is
    routed and a crime prompt is mostly instruction (median post: 10 words);
  - the leading BOS is dropped, because mergekit (moe/router.py,
    tokenize_prompts) prepends it and tokenises with add_special_tokens=False.

With --whole K each expert also gets the WHOLE prompts (ending in [/INST]) of
its first K sampled rows, beside their cuts. A whole prompt is otherwise just one
of a row's ~30 (crime) or ~530 (XSum) cut points, so at --n 64 two crime prompts
and no XSum one ended in [/INST]. That position produces the answer's first
token, and in 86's MoE it went to the crime expert in every one of layers 0-7
for XSum inputs (87's probe). The cuts themselves are unchanged by --whole.

With --negatives each expert also gets the other task's inputs, cut likewise
from different rows, as negative prompts. With two experts and top-1 routing
the choice depends only on the difference of the two router rows, so this
mainly removes the component both tasks share; the rows come out nearly
opposite, which mergekit reports as degenerate, and for two experts that is
harmless.

Usage, from merge/ (evaluate's environment has what this needs):
    uv run --project ../evaluate python make_mistral_data_router_config.py \
        --tokenizer BASE_DIR --base-model BASE_DIR --xsum-expert XS --crime-expert CR \
        --crime ../../datasets/crime_dataset/train --xsum ../../datasets/xsum_dataset/train \
        --n 64 --seed 0 [--whole K] [--negatives] --output moe.yml

RECORD of the runs behind results/isambard/mistral-mlp-*.json: run by
isambard/86_mistral_mlp_moe_data.slurm and 88_mistral_mlp_moe_whole.slurm on
branch mistral-mlp-moe (8650773), whose layout the paths follow, not main's.
It loads evaluate/evaluate.py from that branch, which main does not have.
"""

import argparse
import importlib.util
import random
import re
from pathlib import Path

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

EVALUATE = Path(__file__).resolve().parent.parent / "evaluate" / "evaluate.py"


def crime_build_messages():
    """build_messages from evaluate/evaluate.py, loaded by path, as mistral_router_probe.py."""
    spec = importlib.util.spec_from_file_location("crime_evaluate", EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_messages


def rendered(tokenizer, task, row, build_messages):
    """The prompt as the evaluator renders it, and where its user content starts."""
    if task == "crime":
        content = build_messages(row["posts_name"])[0]["content"]
    else:
        content = row["prompt"]
    text = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    start = text.index(content)
    return text, start, start + len(content)


def cut_prompts(tokenizer, task, rows, n, rng, build_messages, whole=0):
    """n rows, each cut at a random word; the first `whole` rows also whole.
    `whole` draws nothing from rng, so the cuts do not depend on it."""
    bos = tokenizer.bos_token or ""
    prompts = []
    for k, i in enumerate(rng.sample(range(len(rows)), n)):
        text, start, end = rendered(tokenizer, task, rows[i], build_messages)
        if not text.startswith(bos):
            raise ValueError("the chat template does not start with the BOS token")
        cuts = [start + m.end() for m in re.finditer(r"\S+", text[start:end])] + [len(text)]
        prompts.append(text[len(bos):rng.choice(cuts)])
        if k < whole:
            prompts.append(text[len(bos):])
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True, help="base model directory, for its tokenizer")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--xsum-expert", required=True, help="materialised XSum expert")
    parser.add_argument("--crime-expert", required=True, help="materialised crime expert")
    parser.add_argument("--crime", required=True, help="crime training set (load_from_disk)")
    parser.add_argument("--xsum", required=True, help="XSum training set (load_from_disk)")
    parser.add_argument("--n", type=int, default=64, help="prompts per expert (and negatives)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--whole", type=int, default=0, help="whole prompts per expert, from its first sampled rows")
    parser.add_argument("--negatives", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    build_messages = crime_build_messages()
    rows = {"crime": load_from_disk(args.crime), "xsum": load_from_disk(args.xsum)}
    rng = random.Random(args.seed)
    # Positives first for both tasks, then negatives, so that adding
    # --negatives leaves the positives unchanged. sample() draws distinct rows
    # within a call; a negative may reuse a row the other expert's positives
    # used, cut at a different point.
    if not 0 <= args.whole <= args.n:
        parser.error("--whole must be between 0 and --n")
    positive = {t: cut_prompts(tokenizer, t, rows[t], args.n, rng, build_messages, args.whole)
                for t in ("xsum", "crime")}
    negative = {t: cut_prompts(tokenizer, t, rows[t], args.n, rng, build_messages) for t in ("xsum", "crime")} \
        if args.negatives else None

    experts = []
    for task, model, other in (("xsum", args.xsum_expert, "crime"), ("crime", args.crime_expert, "xsum")):
        expert = {"source_model": model, "positive_prompts": positive[task]}
        if negative:
            expert["negative_prompts"] = negative[other]
        experts.append(expert)
    config = {"base_model": args.base_model, "gate_mode": "hidden", "experts_per_token": 1,
              "dtype": "bfloat16", "experts": experts}
    header = (f"# Generated by merge/make_mistral_data_router_config.py --n {args.n} --seed {args.seed}"
              f"{f' --whole {args.whole}' if args.whole else ''}{' --negatives' if args.negatives else ''}:"
              " routing prompts cut from the training\n"
              "# inputs as the evaluators render them. Expert order: 0 = XSum, 1 = crime.\n")
    # allow_unicode=False: written raw, some characters (U+0085, for one) are
    # read back by yaml.safe_load as something else, and the prompt would no
    # longer be a prefix of the evaluator's input. Escaped, they round-trip.
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=False, width=10**6)
    if yaml.safe_load(text) != config:
        raise RuntimeError("the config does not read back as written")
    Path(args.output).write_text(header + text)
    for task in ("xsum", "crime"):
        lengths = sorted(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in positive[task])
        print(f"{task}: {len(lengths)} prompts, {lengths[0]}-{lengths[-1]} tokens (median {lengths[len(lengths) // 2]}), "
              f"{sum(p.endswith('[/INST]') for p in positive[task])} whole")


if __name__ == "__main__":
    main()
