# Writes the merge configs for a task split into N parts, one per method:
#   merge/<task>/<task>_merge_<N>_<method>_config.yaml
# and prints the sbatch commands that run them.
#
# Usage, from merge/:
#   python make_merge_configs.py <task> <N> [--methods m1 m2 ...] [--force]
#   e.g. python make_merge_configs.py crime 4
#        python make_merge_configs.py xsum 8 --methods karcher multislerp
#
# The parts are read from ../models/<task>/split_<N>/ under the names
# train.sbatch / train_kd.sbatch give them, and the merged models go next to
# them. Parameters match the hand-written 2-way configs (weight 1/N, ties
# density 0.5, dare_ties density 0.3), so N = 2, 4, 8 differ only in N.
#
# Methods that take any number of models (linear, ties, dare_ties,
# task_arithmetic, model_stock, karcher, multislerp) get one config. slerp and
# arcee_fusion take exactly two, so for them the config is a binary tree of
# pairwise merges, (1,2) (3,4) ... then their results pairwise, until one model
# is left. It is a multi-document YAML for mergekit-multi: each intermediate
# stage is named <merged name>_parts_<a>_<b> (the range of parts it covers) and
# merge.sbatch saves it to ../models/<task>/split_<N>/intermediate/. Stages are
# written bottom-up, the order merge.sbatch writes their provenance in. With
# t = 0.5 at every level each part ends up with weight 1/N, so N must be a
# power of two. arcee_fusion is not symmetric (it keeps base_model and adds the
# other model's important deltas); the lower-numbered side is always the base.
import argparse
import sys
from pathlib import Path

BASE = "google/gemma-3-4b-it"
TASKS = {
    # Crime parts are LoRA adapters, merged via mergekit's <base>+<adapter> syntax.
    "crime": {
        "part": BASE + "+../models/crime/split_{n}/gemma3_4b_crime_{i}_of_{n}",
        "merged": "gemma3_4b_crime_merge_{n}_{method}",
        "lora": True,
    },
    "xsum": {
        "part": "../models/xsum/split_{n}/gemma3_4b_xsum_{i}_of_{n}_short",
        "merged": "gemma3_4b_xsum_short_merge_{n}_{method}",
        "lora": False,
    },
}
NWAY = {
    # method: (comment, base_model?, per-model parameters as "key: value" lines)
    "linear": ("Linear: weighted average of the raw weights.", False, ["weight: {w}"]),
    "task_arithmetic": ("Task Arithmetic (https://arxiv.org/abs/2212.04089): weighted sum of\n"
                        "# the (model - base) deltas.", True, ["weight: {w}"]),
    "ties": ("TIES (https://arxiv.org/abs/2306.01708): trims each delta to its top\n"
             "# `density` by magnitude, then keeps only the majority-sign contributions.",
             True, ["weight: {w}", "density: 0.5"]),
    "dare_ties": ("DARE + TIES (https://arxiv.org/abs/2311.03099): randomly keeps `density`\n"
                  "# of each delta, rescaled, then applies TIES' sign consensus.",
                  True, ["weight: {w}", "density: 0.3"]),
    "model_stock": ("Model Stock (https://arxiv.org/abs/2403.19522): interpolates towards the\n"
                    "# base by an amount estimated from the angles between the deltas.", True, []),
    "karcher": ("Karcher mean: the geometric mean of the models on the sphere, the\n"
                "# N-way counterpart of slerp at its midpoint. No weights.", False, []),
    "multislerp": ("Multi-SLERP: spherical interpolation of N models with weights, the\n"
                   "# N-way generalisation of slerp.", False, ["weight: {w}"]),
}
PAIRWISE = {
    "slerp": "SLERP: spherical interpolation of two models at t = 0.5.",
    "arcee_fusion": "Arcee Fusion: keeps base_model and adds the other model's deltas whose\n"
                    "# importance (KL shift vs size) is above a computed threshold.",
}
DEFAULT_METHODS = ["linear", "ties", "dare_ties", "task_arithmetic", "model_stock",
                   "slerp", "arcee_fusion"]
FOOTER = "tokenizer_source: union\ndtype: bfloat16\n"


def nway_config(method, parts):
    comment, uses_base, params = NWAY[method]
    w = round(1 / len(parts), 6)
    lines = [f"# {comment}", f"merge_method: {method}"]
    if uses_base:
        lines.append(f'base_model: "{BASE}"')
    lines.append("models:")
    for p in parts:
        lines.append(f'  - model: "{p}"')
        if params:
            lines.append("    parameters:")
            lines += [f"      {q.format(w=w)}" for q in params]
    return "\n".join(lines) + "\n" + FOOTER


def pair_doc(method, name, a, b):
    lines = [f"name: {name}"] if name else []
    lines += [f"merge_method: {method}", f'base_model: "{a}"', "models:", f'  - model: "{b}"']
    if method == "slerp":
        lines += ["parameters:", "  t: 0.5"]
    return "\n".join(lines) + "\n" + FOOTER


def tree_config(method, parts, merged):
    """Pairwise merges, bottom-up; the last document (unnamed) is the final model."""
    level = [(i + 1, i + 1, p) for i, p in enumerate(parts)]  # (first part, last part, ref)
    docs = []
    while len(level) > 1:
        nxt = []
        for (a0, _, a), (_, b1, b) in zip(level[::2], level[1::2]):
            name = None if len(level) == 2 else f"{merged}_parts_{a0}_{b1}"
            docs.append(pair_doc(method, name, a, b))
            nxt.append((a0, b1, name))
        level = nxt
    head = (f"# {PAIRWISE[method]}\n# Takes exactly two models, so the {len(parts)} parts are merged pairwise,\n"
            "# bottom-up, with mergekit-multi (see make_merge_configs.py).\n")
    return head + "---\n".join(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("n", type=int)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=[*NWAY, *PAIRWISE])
    ap.add_argument("--force", action="store_true", help="overwrite existing configs")
    args = ap.parse_args()
    n, spec = args.n, TASKS[args.task]
    if n < 2 or n & (n - 1):
        sys.exit("N must be a power of two (2, 4, 8, ...)")
    parts = [spec["part"].format(n=n, i=i) for i in range(1, n + 1)]
    out_models = f"../models/{args.task}/split_{n}"
    cache = f" --lora-merge-cache {out_models}/.lora_merge_cache" if spec["lora"] else ""
    commands = []
    for method in args.methods:
        merged = spec["merged"].format(n=n, method=method)
        text = tree_config(method, parts, merged) if method in PAIRWISE else nway_config(method, parts)
        path = Path(args.task, f"{args.task}_merge_{n}_{method}_config.yaml")
        if path.exists() and not args.force:
            print(f"# {path} exists, left as it is (--force to overwrite)", file=sys.stderr)
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
            print(f"# wrote {path}", file=sys.stderr)
        commands.append(f"merge.sbatch {path} {out_models}/{merged}{cache}")
    # The LoRA cache is shared, so the first merge fills it before the rest start.
    print("first=$(sbatch --parsable " + commands[0] + ")")
    dep = "--dependency=afterok:$first " if spec["lora"] else ""
    for c in commands[1:]:
        print(f"sbatch {dep}{c}")


if __name__ == "__main__":
    main()
