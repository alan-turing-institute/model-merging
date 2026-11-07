# Writes the merge configs for a task split into N parts, one per method:
#   merge/<task>/<task>_merge_<N>_<method>_config.yaml
# and prints the sbatch commands that run them.
#
# Usage, from merge/:
#   python make_merge_configs.py <task> <N> [--methods m1 m2 ...] [--seed S] [--force]
#   e.g. python make_merge_configs.py crime 4
#        python make_merge_configs.py xsum 8 --methods karcher multislerp
#        python make_merge_configs.py crime 4 --seed 1
#        python make_merge_configs.py clinc150_domain 2
#        python make_merge_configs.py wine 2 --seed 1 --base google/gemma-3-1b-it
#
# The parts are read from ../models/<task>/split_<N>/ under the names
# train.sbatch / train_kd.sbatch give them, and the merged models go next to
# them. xsum_gold is the XSum students trained on the gold summaries, and
# xsum_long / xsum_gold_long those with the long student prompt; their models
# and configs are in the xsum folders, named with the task. With --seed S (a repeat whose parts were trained with --seed S) both
# are under ../models/<task>/seed_<S>/split_<N>/ instead, the configs go to
# <task>/seed_<S>/, and the printed commands merge with --seed S.
# --base (default google/gemma-3-4b-it) is the model the parts were trained
# from: its size is in every model name (gemma3_<size>_...), and for any other
# base the models and configs go in <task>_<size> folders (e.g. crime_1b), so
# that a repeat at another size never shares a directory with the 4B one.
# Parameters match the hand-written 2-way configs (weight 1/N, ties density
# 0.5, dare_ties density 0.3), so N = 2, 4, 8 differ only in N.
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
import re
import sys
from pathlib import Path

BASE = "google/gemma-3-4b-it"
TASKS = {
    # Crime parts are LoRA adapters, merged via mergekit's <base>+<adapter> syntax.
    # {dir} is the split's model dir, ../models/<task>[_<size>][/seed_<S>]/split_<N>;
    # {base} is the base model and {size} its size (4b, 1b).
    "crime": {
        "part": "{base}+{dir}/gemma3_{size}_crime_{i}_of_{n}",
        "merged": "gemma3_{size}_crime_merge_{n}_{method}",
        "lora": True,
    },
    # Wine (points rating) and German->English medical/legal translation parts
    # are LoRA adapters as well, trained by ../train/submit_lora_seed.sh.
    **{
        task: {
            "part": f"{{base}}+{{dir}}/gemma3_{{size}}_{task}_{{i}}_of_{{n}}",
            "merged": f"gemma3_{{size}}_{task}_merge_{{n}}_{{method}}",
            "lora": True,
        }
        for task in ("wine", "mt_domain_de_en")
    },
    # CLINC150 parts are LoRA adapters too: random halves from ../split_dataset.py,
    # and (clinc150_domain) the domain experts, five domains each, from
    # ../train/prepare_data_clinc150.py; both in the clinc150 folders.
    "clinc150": {
        "part": "{base}+{dir}/gemma3_{size}_clinc150_{i}_of_{n}",
        "merged": "gemma3_{size}_clinc150_merge_{n}_{method}",
        "lora": True,
    },
    "clinc150_domain": {
        "folder": "clinc150",
        "part": "{base}+{dir}/gemma3_{size}_clinc150_domain_{i}_of_{n}",
        "merged": "gemma3_{size}_clinc150_domain_merge_{n}_{method}",
        "lora": True,
    },
    # XSum students, all kept in the xsum folders ("folder" overrides the task
    # name in the paths): trained on the teacher's summaries or the gold ones
    # (--targets gold, _gold), with the short or the long student prompt (_long).
    **{
        f"xsum{gold}{'' if prompt == 'short' else '_long'}": {
            "folder": "xsum",
            "part": f"{{dir}}/gemma3_{{size}}_xsum{gold}_{{i}}_of_{{n}}_{prompt}",
            "merged": f"gemma3_{{size}}_xsum{gold}_{prompt}_merge_{{n}}_{{method}}",
            "lora": False,
        }
        for gold in ("", "_gold") for prompt in ("short", "long")
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
# LoRA parts on another base (Gemma 3 1B) are merged without a tokenizer source:
# the 1B tokenizer has one more token than the model's 262,144 embedding rows,
# which mergekit's tokenizer merge cannot place (KeyError: 262144), and every
# part shares the base's tokenizer and embedding anyway, so mergekit can merge
# the tensors as they are and copy the base's tokenizer. The XSum students
# keep the union: Axolotl resized their embeddings to the tokenizer.
FOOTER_SHARED = "dtype: bfloat16\n"


def nway_config(method, parts, base=BASE, footer=FOOTER):
    comment, uses_base, params = NWAY[method]
    w = round(1 / len(parts), 6)
    lines = [f"# {comment}", f"merge_method: {method}"]
    if uses_base:
        lines.append(f'base_model: "{base}"')
    lines.append("models:")
    for p in parts:
        lines.append(f'  - model: "{p}"')
        if params:
            lines.append("    parameters:")
            lines += [f"      {q.format(w=w)}" for q in params]
    return "\n".join(lines) + "\n" + footer


def pair_doc(method, name, a, b, footer=FOOTER):
    lines = [f"name: {name}"] if name else []
    lines += [f"merge_method: {method}", f'base_model: "{a}"', "models:", f'  - model: "{b}"']
    if method == "slerp":
        lines += ["parameters:", "  t: 0.5"]
    return "\n".join(lines) + "\n" + footer


def tree_config(method, parts, merged, footer=FOOTER):
    """Pairwise merges, bottom-up; the last document (unnamed) is the final model."""
    level = [(i + 1, i + 1, p) for i, p in enumerate(parts)]  # (first part, last part, ref)
    docs = []
    while len(level) > 1:
        nxt = []
        for (a0, _, a), (_, b1, b) in zip(level[::2], level[1::2]):
            name = None if len(level) == 2 else f"{merged}_parts_{a0}_{b1}"
            docs.append(pair_doc(method, name, a, b, footer))
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
    ap.add_argument("--seed", type=int,
                    help="a repeat trained with this seed: reads and writes ../models/<task>/seed_<S>/, "
                         "writes the configs to <task>/seed_<S>/, and merges with --seed <S>")
    ap.add_argument("--base", default=BASE,
                    help="the model the parts were trained from (default %(default)s); another size "
                         "uses <task>_<size> folders")
    args = ap.parse_args()
    n, spec = args.n, TASKS[args.task]
    if n < 2 or n & (n - 1):
        sys.exit("N must be a power of two (2, 4, 8, ...)")
    size = re.fullmatch(r"google/gemma-3-(\d+b)-it", args.base)
    if not size:
        sys.exit(f"--base must be google/gemma-3-<size>-it, not {args.base}")
    size = size.group(1)
    seed_dir = f"seed_{args.seed}" if args.seed is not None else ""
    folder = spec.get("folder", args.task) + ("" if args.base == BASE else f"_{size}")
    out_models = str(Path("../models", folder, seed_dir, f"split_{n}"))
    parts = [spec["part"].format(base=args.base, size=size, dir=out_models, n=n, i=i) for i in range(1, n + 1)]
    cache = f" --lora-merge-cache {out_models}/.lora_merge_cache" if spec["lora"] else ""
    footer = FOOTER_SHARED if spec["lora"] and args.base != BASE else FOOTER
    seed = f" --seed {args.seed}" if args.seed is not None else ""
    commands = []
    for method in args.methods:
        merged = spec["merged"].format(size=size, n=n, method=method)
        text = (tree_config(method, parts, merged, footer) if method in PAIRWISE
                else nway_config(method, parts, args.base, footer))
        path = Path(folder, seed_dir, f"{args.task}_merge_{n}_{method}_config.yaml")
        if path.exists() and not args.force:
            print(f"# {path} exists, left as it is (--force to overwrite)", file=sys.stderr)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"# wrote {path}", file=sys.stderr)
        commands.append(f"merge.sbatch {path} {out_models}/{merged}{seed}{cache}")
    # The LoRA cache is shared, so the first merge fills it before the rest start.
    print("first=$(sbatch --parsable " + commands[0] + ")")
    dep = "--dependency=afterok:$first " if spec["lora"] else ""
    for c in commands[1:]:
        print(f"sbatch {dep}{c}")


if __name__ == "__main__":
    main()
