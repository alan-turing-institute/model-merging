"""Which expert a task MoE's router picks, on the inputs it was evaluated on.

The task MoEs (make_moe_configs.py) are Mixtral models with four experts, one
per task (crime, wine, CLINC150, XSum), and experts_per_token 1: at every layer
each token goes through exactly one expert's MLP, the one its router scores
highest. Routing is per token and per layer, not per query. This measures it:

  home share     the share of (token, layer) routing decisions that go to the
                 task's own expert. Near 1: inputs run through their own
                 expert; near 1/4: no better than chance; one expert near 1 for
                 every task: collapsed onto that expert.
  per layer      the home share at each layer.
  confusion      for each task, the share of decisions going to each expert.
  vs score       per sample, its answer's home share against its score: do the
                 samples that route badly score badly?

Each is reported for three sets of tokens: the prompt, its last token (whose
state produces the first answer token) and the answer. The answer is the
MoE's own, from its evaluation, appended to the prompt: with greedy decoding a
token's hidden state, and so its routing, is the one it had when generated.

Inputs come from the MoE's result JSONs (evaluate/export_results.py), which
hold each sample's prompt (XSum's in the shared prompts file), the MoE's answer
and its score, so the prompts are exactly those evaluated. They are rendered as
Inspect's hf provider does: the chat template to text, then tokenised with the
tokenizer's defaults. The answer is tokenised on its own and appended (no end
of turn: that token is generated but never fed back in), so a re-tokenised
answer can differ from the generated ids by a token or so.

Usage, from merge/ (in evaluate/'s environment; see moe_router_probe.sbatch):
  uv run --project ../evaluate python moe_router_probe.py MOE_DIR RESULTS_DIR \
      --limit 200 --output routing.json
where RESULTS_DIR holds the MoE's crime/wine/clinc150/xsum JSONs, e.g.
../results/task_merging/mistral/seed_1/moe.

Based on mistral_router_probe.py (2-expert Mistral MoE, crime and XSum) by
mariopi3, in commit b8937e0.
"""

import argparse
import json
import random
import re
import statistics as st
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = ["crime", "wine", "clinc150", "xsum"]
# Each task's per-sample score in its result JSON: scorer, value key, higher is better?
SCORE = {
    "crime": ("crime_label", "accuracy", True),
    "wine": ("points_error", "abs_error", False),
    "clinc150": ("intent_match", "accuracy", True),
    "xsum": ("rouge", "rouge1", True),
}
SEGMENTS = ["prompt", "last_prompt_token", "answer"]


def expert_tasks(moe_dir):
    """The task of each expert, in router order, from the MoE's mergekit config."""
    cfg = yaml.safe_load(Path(moe_dir, "mergekit_moe_config.yml").read_text())
    tasks = []
    for e in cfg["experts"]:
        m = re.search(r"models/([a-z0-9]+)/seed_\d", e["source_model"])
        if not m or m.group(1) not in TASKS:
            raise ValueError(f"cannot tell the task of expert {e['source_model']}")
        tasks.append(m.group(1))
    # A router replaced by set_moe_router.py is described by router.json.
    router = Path(moe_dir, "router.json")
    gate_mode = json.loads(router.read_text())["gate_mode"] if router.exists() else cfg.get("gate_mode")
    return tasks, gate_mode


def load_samples(results_dir, task, limit, seed):
    """A fixed random sample of the task's evaluated samples, with prompts."""
    path = Path(results_dir, f"{task}.json")
    d = json.loads(path.read_text())
    shared = json.loads((path.parent / d["prompts_file"]).read_text())["prompts"] if "prompts_file" in d else {}
    samples = d["samples"]
    keep = sorted(random.Random(seed).sample(range(len(samples)), min(limit, len(samples))))
    scorer, key, _ = SCORE[task]
    out = []
    for i in keep:
        s = samples[i]
        prompt = s.get("prompt") or shared[str(s["id"])]
        value = ((s["scores"].get(scorer) or {}).get("value") or {}).get(key)
        out.append({"id": s["id"], "prompt": prompt, "output": s["output"] or "", "score": value})
    return out, d["source_log"]


@torch.no_grad()
def route(model, ids):
    """Per layer and token, the chosen expert and the router's probabilities:
    top1 (layers, tokens) and probs (layers, tokens, experts)."""
    out = model(input_ids=torch.tensor([ids], device=model.device), output_router_logits=True)
    if not out.router_logits:
        raise RuntimeError("the model returned no router logits: is this a MoE?")
    logits = torch.stack([lg.float().reshape(len(ids), -1) for lg in out.router_logits])
    return logits.argmax(-1).cpu(), logits.softmax(-1).cpu()


class Tally:
    """Routing decisions of one set of tokens: counts per layer and expert."""

    def __init__(self, layers, experts):
        self.counts = torch.zeros(layers, experts, dtype=torch.long)
        self.prob = torch.zeros(experts, dtype=torch.double)

    def add(self, top1, probs):
        if top1.shape[1]:
            self.counts += torch.nn.functional.one_hot(top1, self.counts.shape[1]).sum(1)
            self.prob += probs.double().sum((0, 1))

    def summary(self, home):
        n = self.counts.sum(1, keepdim=True).clamp(min=1).double()
        share = self.counts.double() / n  # (layers, experts)
        tokens = int(self.counts[0].sum())
        return {
            "tokens": tokens,
            "home_share": share[:, home].mean().item(),
            "home_share_per_layer": [round(x, 4) for x in share[:, home].tolist()],
            "expert_share": [round(x, 4) for x in share.mean(0).tolist()],
            "mean_home_prob": (self.prob[home] / max(tokens * share.shape[0], 1)).item(),
        }


def probe_task(model, tokenizer, task, samples, home, n_layers, n_experts):
    tallies = {seg: Tally(n_layers, n_experts) for seg in SEGMENTS}
    per_sample = []
    for s in samples:
        text = tokenizer.apply_chat_template(s["prompt"], add_generation_prompt=True, tokenize=False)
        ids = tokenizer(text)["input_ids"]  # as Inspect's hf provider: tokenizer defaults
        answer = tokenizer(s["output"], add_special_tokens=False)["input_ids"]
        top1, probs = route(model, ids + answer)
        n = len(ids)
        parts = {"prompt": (top1[:, :n], probs[:, :n]), "last_prompt_token": (top1[:, n - 1:n], probs[:, n - 1:n]),
                 "answer": (top1[:, n:], probs[:, n:])}
        for seg, (t, p) in parts.items():
            tallies[seg].add(t, p)
        a = top1[:, n:]
        per_sample.append({
            "id": s["id"], "prompt_tokens": n, "answer_tokens": len(answer), "score": s["score"],
            "prompt_home": round((top1[:, :n] == home).float().mean().item(), 4),
            "answer_home": round((a == home).float().mean().item(), 4) if a.numel() else None,
            "answer_expert": torch.bincount(a.flatten(), minlength=n_experts).argmax().item() if a.numel() else None,
        })
    result = {seg: tallies[seg].summary(home) for seg in SEGMENTS}
    result["samples"] = len(samples)
    result["answers_mostly_home"] = st.mean(p["answer_home"] > 0.5 for p in per_sample if p["answer_home"] is not None)
    result["score_by_routing"] = score_by_routing(task, per_sample)
    result["per_sample"] = per_sample
    return result


def score_by_routing(task, per_sample):
    """Mean score of samples whose answer mostly went to the home expert, and
    of the rest, and the correlation of answer home share with score."""
    rows = [(p["answer_home"], p["score"]) for p in per_sample if p["answer_home"] is not None and p["score"] is not None]
    home = [s for h, s in rows if h > 0.5]
    away = [s for h, s in rows if h <= 0.5]
    corr = None
    if len(rows) > 2 and len({h for h, _ in rows}) > 1 and len({s for _, s in rows}) > 1:
        corr = st.correlation([h for h, _ in rows], [s for _, s in rows])
    _, key, higher = SCORE[task]
    return {"metric": key, "higher_is_better": higher,
            "mostly_home": {"n": len(home), "mean": st.mean(home) if home else None},
            "mostly_elsewhere": {"n": len(away), "mean": st.mean(away) if away else None},
            "correlation_answer_home_share_vs_score": corr}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("moe", help="the MoE model directory")
    ap.add_argument("results", help="its result JSONs: <results>/{crime,wine,clinc150,xsum}.json")
    ap.add_argument("--limit", type=int, default=200, help="samples per task (a fixed random sample)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    experts, gate_mode = expert_tasks(args.moe)
    tokenizer = AutoTokenizer.from_pretrained(args.moe)
    model = AutoModelForCausalLM.from_pretrained(args.moe, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    n_layers, n_experts = model.config.num_hidden_layers, model.config.num_local_experts
    assert n_experts == len(experts), (n_experts, experts)

    result = {"model": str(Path(args.moe).resolve()), "gate_mode": gate_mode, "experts": experts,
              "experts_per_token": model.config.num_experts_per_tok, "layers": n_layers,
              "limit": args.limit, "sample_seed": args.sample_seed, "tasks": {}}
    print(f"{Path(args.moe).name}: gate_mode {gate_mode}, experts {experts}")
    for task in TASKS:
        samples, source = load_samples(args.results, task, args.limit, args.sample_seed)
        r = probe_task(model, tokenizer, task, samples, experts.index(task), n_layers, n_experts)
        r["source_json"] = str(Path(args.results, f"{task}.json"))
        r["source_log"] = source
        result["tasks"][task] = r
        sb = r["score_by_routing"]
        print(f"  {task:9s} home share: prompt {r['prompt']['home_share']:.3f}, last prompt token "
              f"{r['last_prompt_token']['home_share']:.3f}, answer {r['answer']['home_share']:.3f}; "
              f"answer tokens by expert {r['answer']['expert_share']}; {sb['metric']} mostly home "
              f"{sb['mostly_home']['mean']} (n={sb['mostly_home']['n']}) vs elsewhere "
              f"{sb['mostly_elsewhere']['mean']} (n={sb['mostly_elsewhere']['n']})")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
