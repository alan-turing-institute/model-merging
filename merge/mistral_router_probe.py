"""Which expert the Mistral MoE's router picks, on the prompts it is scored on.

gate_mode: hidden builds each layer's gate from hidden states over a handful of
prompts and never trains it. With experts_per_token 1 each token passes through
one expert's MLP per layer, so the share of tokens sent to the expert trained on
that task says how much of the MoE's score is down to routing:

  near 1 on both tasks     the MoE runs each input through its own expert and
                           should score close to its parents
  near 0.5                 the gate is noise; the MoE mixes the experts token
                           by token
  near 0 or 1 everywhere   collapsed to one expert; the MoE is that expert

Prompts are rendered as the evaluators render them: crime through
evaluate/evaluate.py's build_messages, XSum from the dataset's `prompt` column,
each as one user turn through the tokenizer's chat template with the generation
prompt. Expert order is moe_mistral_mlp_2way.yml's: 0 = XSum, 1 = crime.

ANSWERS. The prompt is only part of what is routed: every generated token is
fed back in and routed too. With --own-crime and --own-xsum (the MoE's own
evaluation JSONs) the probe also routes the answers the MoE gave, forced as the
assistant turn. With greedy decoding a forced token's hidden state is the one it
had when generated, so this is the routing of generation itself. --reference
routes the reference answers as well. Answer tokens are those after the
prompt, without the closing </s>, which is generated but never fed back in.
evaluate.py keeps only the parsed crime label, so the MoE's own crime answers
are forced as "crime" / "not_crime" (unparsed ones are skipped). XSum keeps the
text. Both are re-tokenised, so a forced answer can differ from the generated
ids by a token or so.

Usage, from merge/ (evaluate's environment has everything this needs):

    uv run --project ../evaluate python mistral_router_probe.py MOE_DIR \
        --crime ../../datasets/crime_dataset/test \
        --xsum ../../datasets/xsum_dataset/test --limit 200 --output routing.json \
        [--own-crime MOE-crime.json --own-xsum MOE-xsum.json] [--reference]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

EVALUATE = Path(__file__).resolve().parent.parent / "evaluate" / "evaluate.py"
HOME_EXPERT = {"xsum": 0, "crime": 1}
CRIME_LABELS = {1: "crime", 0: "not_crime"}


def crime_build_messages():
    """build_messages from evaluate/evaluate.py, loaded by path: evaluate/ is a
    directory of scripts, and `evaluate` is also the name of a pip package."""
    spec = importlib.util.spec_from_file_location("crime_evaluate", EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_messages


def user_turn(task, row, build_messages):
    if task == "crime":
        return build_messages(row["posts_name"])
    return [{"role": "user", "content": row["prompt"]}]


def prompt_ids(tokenizer, task, row, build_messages):
    """The prompt's token ids, as the evaluator computes them."""
    if task == "crime":
        return tokenizer.apply_chat_template(user_turn(task, row, build_messages), tokenize=True,
                                             add_generation_prompt=True, return_dict=True)["input_ids"]
    # As evaluate_summarisation.py: render to text, then tokenise without a
    # second BOS.
    text = tokenizer.apply_chat_template(user_turn(task, row, build_messages), tokenize=False,
                                         add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def conversation_ids(tokenizer, task, row, build_messages, answer):
    messages = user_turn(task, row, build_messages) + [{"role": "assistant", "content": answer}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def top1(model, ids):
    """Per layer, each token's top-1 expert: a (layers, tokens) tensor."""
    out = model(input_ids=torch.tensor([ids], device=model.device), output_router_logits=True)
    if not out.router_logits:
        raise RuntimeError("the model returned no router logits - is this a MoE?")
    return torch.stack([logits.float().argmax(-1) for logits in out.router_logits]).cpu()


class Tally:
    """Share of tokens routed to one expert, per layer, pooled over sequences."""

    def __init__(self, home):
        self.home, self.hits, self.tokens, self.per_sequence = home, None, 0, []

    def add(self, top):
        if top.shape[1] == 0:
            return
        home = top == self.home
        self.hits = home.sum(dim=1) if self.hits is None else self.hits + home.sum(dim=1)
        self.tokens += top.shape[1]
        self.per_sequence.append(home.float().mean().item())

    def summary(self):
        per_layer = (self.hits.double() / self.tokens).tolist()
        return {"sequences": len(self.per_sequence), "tokens": self.tokens,
                "home_share": sum(per_layer) / len(per_layer), "home_share_per_layer": per_layer,
                "mostly_home": sum(p > 0.5 for p in self.per_sequence) / len(self.per_sequence)}


def answers_from(task, rows, own_json):
    """The MoE's own answers, one per row (None where there is none)."""
    predictions = json.load(open(own_json))["predictions"]
    if len(predictions) != len(rows):
        raise ValueError(f"{own_json} has {len(predictions)} predictions for {len(rows)} rows")
    if task == "crime":
        return [CRIME_LABELS.get(p) for p in predictions]
    return [p if p else None for p in predictions]


def reference_answers(task, rows):
    if task == "crime":
        return [r["messages"][1]["content"] for r in rows]
    return list(rows["summary"])


def routing(model, tokenizer, task, rows, build_messages, answers):
    """Prompt routing as before, plus the routing of each set of forced answers.

    The prompt's last token ([/INST]) is also tallied on its own: its hidden
    state produces the answer's first token, and it is 1 of ~470 XSum prompt
    tokens, so the pooled prompt share cannot show how it is routed."""
    home = HOME_EXPERT[task]
    prompt, last = Tally(home), Tally(home)
    forced = {name: Tally(home) for name in answers}
    mismatched = {name: 0 for name in answers}
    for i, row in enumerate(rows):
        ids = prompt_ids(tokenizer, task, row, build_messages)
        top = top1(model, ids)
        prompt.add(top)
        last.add(top[:, -1:])
        for name, texts in answers.items():
            if texts[i] is None:
                continue
            full = conversation_ids(tokenizer, task, row, build_messages, texts[i])
            if full[:len(ids)] != ids:
                mismatched[name] += 1
                continue
            end = len(full) - 1 if full[-1] == tokenizer.eos_token_id else len(full)
            forced[name].add(top1(model, full)[:, len(ids):end])
    result = prompt.summary()
    result["home_expert"] = home
    result["prompts"] = result.pop("sequences")
    result["prompts_mostly_home"] = result.pop("mostly_home")
    tally = last.summary()
    result["last_prompt_token"] = {"home_share": tally["home_share"],
                                   "home_share_per_layer": tally["home_share_per_layer"]}
    if answers:
        result["answers"] = {name: {**tally.summary(), "prefix_mismatches": mismatched[name]}
                             for name, tally in forced.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="MoE model directory")
    parser.add_argument("--crime", required=True, help="crime test set (load_from_disk)")
    parser.add_argument("--xsum", required=True, help="XSum test set (load_from_disk)")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--own-crime", help="this MoE's crime evaluation JSON, to route its own answers")
    parser.add_argument("--own-xsum", help="this MoE's XSum evaluation JSON, to route its own answers")
    parser.add_argument("--reference", action="store_true", help="also route the reference answers")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.own_crime) != bool(args.own_xsum):
        parser.error("--own-crime and --own-xsum go together")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    build_messages = crime_build_messages()
    result = {"model": args.model, "limit": args.limit}
    for task, path, own in (("crime", args.crime, args.own_crime), ("xsum", args.xsum, args.own_xsum)):
        rows = load_from_disk(path)
        rows = rows.select(range(min(args.limit, len(rows))))
        answers = {}
        if own:
            answers["own"] = answers_from(task, rows, own)
        if args.reference:
            answers["reference"] = reference_answers(task, rows)
        result[task] = r = routing(model, tokenizer, task, rows, build_messages, answers)
        print(f"{task}: {r['home_share']:.3f} of prompt tokens to its own expert "
              f"(layers {min(r['home_share_per_layer']):.3f}-{max(r['home_share_per_layer']):.3f}; "
              f"{r['prompts_mostly_home']:.3f} of prompts mostly), {r['tokens']} tokens")
        t = r["last_prompt_token"]
        print(f"  last prompt token: {t['home_share']:.3f} to its own expert "
              f"(layers {min(t['home_share_per_layer']):.3f}-{max(t['home_share_per_layer']):.3f})")
        for name, a in r.get("answers", {}).items():
            print(f"  {name} answers: {a['home_share']:.3f} of answer tokens to its own expert "
                  f"(layers {min(a['home_share_per_layer']):.3f}-{max(a['home_share_per_layer']):.3f}; "
                  f"{a['mostly_home']:.3f} of answers mostly), {a['tokens']} tokens in {a['sequences']} answers, "
                  f"{a['prefix_mismatches']} skipped as not extending the prompt")
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
