# Routing tools for the task MoE (merge/moe/<model>_task_moe_4_seed_<S>_config.yaml):
# where its tokens go, and four ways to route them better. Every router is
# fitted on training data only; the test sets are used to probe and evaluate.
#
# Usage, from merge/ (moe_routing.sbatch runs it on a GPU node):
#   python moe_routing.py probe <moe_dir> <out.json> [--n 300]
#       routing per layer on each task's test prompts: the share of prompt tokens
#       and of the answer position (the last prompt token) sent to each expert,
#       the prompt-level choice (mean router probability over tokens and
#       layers), and for crime, greedy answers, so routing can be compared
#       between right and wrong answers
#   python moe_routing.py prompts <config.yaml> <out_config.yaml> [--k 16]
#       option 1: a mergekit-moe config whose positive prompts are real chat-
#       formatted training examples, k ending at the answer position and k cut
#       at a random token, since mergekit's hidden gate reads each prompt's
#       last token
#   python moe_routing.py fit <moe_dir> <out_dir> --base <hub id> [--mode logreg|mean]
#       option 2: each layer's router fitted on the base model's hidden states
#       over every token of training sequences, labelled by task
#   python moe_routing.py train <moe_dir> <out_dir> [--lm 1.0] [--route 1.0] [--steps 600]
#       option 3: the routers alone trained on the MoE itself, with a per-token
#       routing loss (task labels) and the answer tokens' loss, which reaches
#       the router through a straight-through top-1 weight
# Option 4, routing per prompt, needs no new model: an MoE that sends every token
# to expert e is expert e's model (the experts differ from the base only in their
# MLPs), so its score is the expert's on every prompt the probe's prompt-level
# choice classifies correctly.
import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = ["crime", "wine", "clinc150", "xsum"]  # the experts' order in the MoE configs
DATA = {"crime": "crime/crime_dataset", "wine": "wine/wine_dataset",
        "clinc150": "clinc150/clinc150_labels_dataset", "xsum": "xsum/xsum_prompts"}
MAX_LEN = 2048


def conversations(task, split):
    """The task's (user turn, answer) pairs, as the experts were trained and evaluated."""
    d = load_from_disk(f"../datasets/{DATA[task]}")[split]
    if task == "xsum":
        return [(r["messages_long"], r["summary"]) for r in d]
    return [(r["messages"][:1], r["messages"][1]["content"]) for r in d]


def encode(tok, user, answer=None):
    """Token ids of the chat-formatted prompt (and answer), and the prompt's length."""
    def ids(messages, **kw):  # the template's text (with its BOS), tokenised as is
        return tok(tok.apply_chat_template(messages, tokenize=False, **kw), add_special_tokens=False)["input_ids"]
    prompt = ids(user, add_generation_prompt=True)
    if answer is None:
        return prompt[:MAX_LEN], min(len(prompt), MAX_LEN)
    full = ids(user + [{"role": "assistant", "content": answer}])
    assert full[:len(prompt)] == prompt, "the chat template does not extend the prompt"
    return full[-MAX_LEN:], max(1, len(prompt) - max(0, len(full) - MAX_LEN))


def load(path, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cuda").eval()
    return tok, model


def routers(model):
    return [layer.mlp.gate for layer in model.model.layers]


def record_routing(model):
    """Hooks that keep each layer's router probabilities for the last forward pass."""
    probs = [None] * len(routers(model))

    def hook(i):
        def f(mod, inp, out):
            probs[i] = torch.softmax(out[0].float(), -1).detach()
        return f
    handles = [g.register_forward_hook(hook(i)) for i, g in enumerate(routers(model))]
    return probs, handles


@torch.no_grad()
def probe(args):
    tok, model = load(args.moe)
    probs, _ = record_routing(model)
    out = {}
    for t, task in enumerate(TASKS):
        convs = conversations(task, "test")
        random.Random(0).shuffle(convs)
        rows = []
        for user, answer in convs[:args.n]:
            ids, n = encode(tok, user)
            model(torch.tensor([ids], device="cuda"))
            p = torch.stack(probs)  # (layers, tokens, experts)
            row = {"tokens_home": p.argmax(-1).eq(t).float().mean(1).tolist(),
                   "answer_expert": p[:, n - 1].argmax(-1).tolist(),
                   "prompt_choice": int(p.mean((0, 1)).argmax()), "length": n}
            if task == "crime":  # the hooks fire again while generating; row is already taken
                g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=6, do_sample=False)
                reply = tok.decode(g[0, n:], skip_special_tokens=True).strip().lower()
                first = re.sub(r"[^a-z_]", "", (reply.split() or [""])[0])
                row.update(target=answer, reply=reply, correct=first == answer)
            rows.append(row)
        L = len(rows[0]["tokens_home"])
        out[task] = {
            "n": len(rows),
            "prompt_choice_home": sum(r["prompt_choice"] == t for r in rows) / len(rows),
            "tokens_home_by_layer": [sum(r["tokens_home"][l] for r in rows) / len(rows) for l in range(L)],
            "answer_home_by_layer": [sum(r["answer_expert"][l] == t for r in rows) / len(rows) for l in range(L)],
            "answer_expert_share": [[sum(r["answer_expert"][l] == e for r in rows) / len(rows) for e in range(4)]
                                    for l in range(L)],
        }
        if task == "crime":
            for ok in (True, False):
                sel = [r for r in rows if r["correct"] == ok]
                out[task]["correct" if ok else "wrong"] = {
                    "n": len(sel),
                    "answer_home_by_layer": [sum(r["answer_expert"][l] == t for r in sel) / max(1, len(sel))
                                             for l in range(L)],
                    "tokens_home": sum(sum(r["tokens_home"]) / L for r in sel) / max(1, len(sel)),
                    "examples": [(r["target"], r["reply"]) for r in sel[:15]]}
        print(task, {k: v for k, v in out[task].items() if not isinstance(v, (list, dict))}, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def prompts(args):
    cfg = yaml.safe_load(open(args.config))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    rng = random.Random(args.seed)
    for t, task in enumerate(TASKS):
        convs = rng.sample(conversations(task, "train"), 2 * args.k)
        texts = []
        for i, (user, answer) in enumerate(convs):
            ids, n = encode(tok, user, answer)
            cut = n if i < args.k else rng.randint(min(8, n), len(ids) - 1)  # answer position, or anywhere
            # mergekit tokenises the text and adds BOS itself
            texts.append(tok.decode(ids[:cut]).removeprefix(tok.bos_token or "").lstrip())
        cfg["experts"][t]["positive_prompts"] = texts
        cfg["experts"][t].pop("negative_prompts", None)
    head = (f"# {Path(args.config).name} with the positive prompts replaced by real chat-formatted training\n"
            f"# examples (moe_routing.py prompts, k = {args.k}, seed {args.seed}): {args.k} per expert end at the\n"
            f"# answer position, {args.k} at a random token, since the hidden gate reads each prompt's last token.\n")
    Path(args.out).write_text(head + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=10**9))


def save_with_gates(args, model, tok, gates, note):
    for g, w in zip(routers(model), gates):
        g.weight.data = w.to(g.weight.device, torch.bfloat16)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    prov = Path(args.moe) / "provenance.json"
    # The MoE's own record (evaluate.sbatch reads its fields), with the new routing noted.
    record = json.loads(prov.read_text()) if prov.exists() else {}
    record["routing"] = {"made_by": "merge/moe_routing.py", **note}
    (Path(args.out) / "provenance.json").write_text(json.dumps(record, indent=1, default=str))


def training_tokens(tok, task, n, rng):
    convs = rng.sample(conversations(task, "train"), n)
    return [encode(tok, user, answer) for user, answer in convs]


def fit(args):
    rng = random.Random(args.seed)
    tok, base = load(args.base)
    layers = base.model.layers
    feats = [[[] for _ in TASKS] for _ in layers]

    # Each sequence gives the same few positions at every layer: the answer
    # position (the last prompt token) and random others, ~2 * tokens / n in all,
    # so the hidden states fit in memory (bf16, on the CPU).
    per_seq = max(2, 2 * args.tokens // args.n)

    def hook(i):
        def f(mod, inp, out):
            h = out[0] if out.dim() == 3 else out
            feats[i][current[0]].append(h[current[1]].to(torch.bfloat16).cpu())
        return f
    current = [0, None]
    handles = [l.post_attention_layernorm.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    with torch.no_grad():
        for t, task in enumerate(TASKS):
            current[0] = t
            for ids, n in training_tokens(tok, task, args.n, rng):
                others = [p for p in rng.sample(range(len(ids)), min(per_seq, len(ids))) if p != n - 1]
                current[1] = torch.tensor([n - 1] + others[:per_seq - 1], device="cuda")
                base(torch.tensor([ids], device="cuda"))
    for h in handles:
        h.remove()
    del base
    torch.cuda.empty_cache()
    gates = []
    for i in range(len(feats)):
        xs, ys = [], []
        for t in range(len(TASKS)):
            x = torch.cat(feats[i][t]).float()
            x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(i))[:args.tokens]]
            xs.append(x)
            ys.append(torch.full((len(x),), t))
        X, y = torch.cat(xs).cuda(), torch.cat(ys).cuda()
        if args.mode == "mean":  # mergekit's hidden rows, from every token of real examples
            W = torch.stack([F.normalize(X[y == t], dim=-1).mean(0) for t in range(len(TASKS))])
        else:  # multinomial logistic regression, no bias (the router has none), balanced classes
            W = torch.zeros(len(TASKS), X.shape[1], device="cuda", requires_grad=True)
            opt = torch.optim.LBFGS([W], max_iter=200, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(X @ W.T, y) + args.l2 * W.pow(2).sum()
                loss.backward()
                return loss
            opt.step(closure)
            W = W.detach()
        acc = (X @ W.T).argmax(-1).eq(y).float().mean().item()
        print(f"layer {i}: training accuracy {acc:.3f}", flush=True)
        gates.append(W)
        feats[i] = None
    tok, model = load(args.moe)
    save_with_gates(args, model, tok, gates, {"option": "fit", "mode": args.mode, "n_per_task": args.n,
                                              "tokens_per_task_per_layer": args.tokens, "l2": args.l2,
                                              "seed": args.seed, "hidden_states_from": args.base})


def train(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tok, model = load(args.moe)
    for p in model.parameters():
        p.requires_grad_(False)
    gates = [torch.nn.Parameter(g.weight.detach().float().clone()) for g in routers(model)]
    route_logits = []
    straight_through = args.lm > 0

    def make_forward(i, mod):
        def forward(hidden_states):
            h = hidden_states.reshape(-1, mod.hidden_dim)
            logits = F.linear(h.float(), gates[i])
            probs = torch.softmax(logits, -1)
            top, idx = torch.topk(probs, mod.top_k, -1)
            # The top-1 weight is 1 after renormalising, so the answer loss cannot
            # reach the router; top / top.detach() is still 1 but passes the gradient.
            scores = top / top.detach() if straight_through else (top / top.sum(-1, keepdim=True)).detach()
            if args.route > 0:
                route_logits.append(F.linear(h.detach().float(), gates[i]))
            return logits, scores.to(hidden_states.dtype), idx
        return forward
    for i, g in enumerate(routers(model)):
        g.forward = make_forward(i, g)
    if straight_through:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    data = {t: training_tokens(tok, task, args.n, rng) for t, task in enumerate(TASKS)}
    val = {t: [encode(tok, u, a) for u, a in rng.sample(conversations(task, "validation"), 40)]
           for t, task in enumerate(TASKS)}
    opt = torch.optim.Adam(gates, lr=args.lr)

    def step_loss(t, ids, n, grad=True):
        route_logits.clear()
        x = torch.tensor([ids], device="cuda")
        with torch.set_grad_enabled(grad):
            out = model(x, use_cache=False)
            loss = torch.zeros((), device="cuda")
            if args.lm > 0:
                logits = out.logits[0, n - 1:-1].float()
                loss = loss + args.lm * F.cross_entropy(logits, x[0, n:])
            if args.route > 0:
                rl = torch.stack(route_logits)  # (layers, tokens, experts)
                loss = loss + args.route * F.cross_entropy(rl.reshape(-1, rl.shape[-1]),
                                                           torch.full((rl.shape[0] * rl.shape[1],), t, device="cuda"))
        home = torch.stack(route_logits).argmax(-1).eq(t).float().mean().item() if route_logits else float("nan")
        return loss, home

    def validate():
        res = {}
        for t in val:
            losses, homes = zip(*[step_loss(t, ids, n, grad=False) for ids, n in val[t]])
            res[TASKS[t]] = (round(sum(l.item() for l in losses) / len(losses), 4), round(sum(homes) / len(homes), 3))
        return res
    print("before:", validate(), flush=True)
    for step in range(1, args.steps + 1):
        t = step % len(TASKS)  # tasks in turn, so each gets the same number of updates
        ids, n = data[t][rng.randrange(len(data[t]))]
        loss, _ = step_loss(t, ids, n)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step}:", validate(), flush=True)
    for g in routers(model):
        del g.forward  # back to the class's forward, with the trained weights
    save_with_gates(args, model, tok, [g.detach() for g in gates],
                    {"option": "train", "lm_weight": args.lm, "route_weight": args.route, "steps": args.steps,
                     "lr": args.lr, "n_per_task": args.n, "seed": args.seed})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("moe"); p.add_argument("out"); p.add_argument("--n", type=int, default=300)
    p = sub.add_parser("prompts"); p.add_argument("config"); p.add_argument("out")
    p.add_argument("--k", type=int, default=16); p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("fit"); p.add_argument("moe"); p.add_argument("out"); p.add_argument("--base", required=True)
    p.add_argument("--mode", choices=["logreg", "mean"], default="logreg"); p.add_argument("--n", type=int, default=300)
    p.add_argument("--tokens", type=int, default=20000); p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("train"); p.add_argument("moe"); p.add_argument("out")
    p.add_argument("--lm", type=float, default=1.0); p.add_argument("--route", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=600); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n", type=int, default=500); p.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    {"probe": probe, "prompts": prompts, "fit": fit, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
