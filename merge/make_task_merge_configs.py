# Writes the task-merging configs: the four MLP-only task experts of one model
# and seed (crime, wine, CLINC150, XSum) merged into one model, one config per
# method:
#   merge/task_merge/seed_<S>/<model>_task_merge_4_<method>_config.yaml
# and prints, one per method, "<method> <config> <output dir>" for the submit
# script (submit_task_merges.sh).
#
# Usage, from merge/:
#   python make_task_merge_configs.py <gemma|mistral|llama> --seed S [--methods m1 m2 ...] [--force]
#
# The experts are the outputs of the MLP_* configs (train/MLP_<task>_<model>.yaml,
# self-distill/MLP_xsum_<model>.yaml), trained with --seed S:
#   ../models/{crime,wine}/seed_<S>/<stem>_<task>_mlp          LoRA, as <base>+<adapter>
#   ../models/clinc150/seed_<S>/<stem>_clinc150_labels_mlp     LoRA, as <base>+<adapter>
#   ../models/xsum/seed_<S>/<stem>_xsum_gold_long_mlp          a full model
# and the merged models go to ../models/task_merge/seed_<S>/.
#
# The methods and their parameters are make_merge_configs.py's (weight 1/N =
# 0.25, ties density 0.5, dare_ties density 0.3), so a task merge differs from
# a data-split merge only in what it merges. slerp and arcee_fusion take two
# models, so they are the same pairwise tree: (crime, wine) and (CLINC150,
# XSum), then the two results; arcee_fusion keeps the first of each pair as
# its base. With t = 0.5 at each level every expert ends up with weight 1/4.
import argparse
import sys
from pathlib import Path

from make_merge_configs import DEFAULT_METHODS, NWAY, PAIRWISE, nway_config, tree_config

MODELS = {
    "gemma": {"base": "google/gemma-3-4b-it", "stem": "gemma3_4b"},
    "mistral": {"base": "mistralai/Mistral-7B-Instruct-v0.3", "stem": "mistral_7b"},
    "llama": {"base": "meta-llama/Llama-3.1-8B-Instruct", "stem": "llama3_8b"},
}
# (task, expert dir name under ../models/<task>/seed_<S>/, LoRA?), in merge order.
EXPERTS = [
    ("crime", "{stem}_crime_mlp", True),
    ("wine", "{stem}_wine_mlp", True),
    ("clinc150", "{stem}_clinc150_labels_mlp", True),
    ("xsum", "{stem}_xsum_gold_long_mlp", False),
]


def experts(model, seed):
    base, stem = MODELS[model]["base"], MODELS[model]["stem"]
    refs = []
    for task, name, lora in EXPERTS:
        path = f"../models/{task}/seed_{seed}/{name.format(stem=stem)}"
        refs.append(f"{base}+{path}" if lora else path)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--seed", type=int, required=True, help="the seed the experts were trained with")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=[*NWAY, *PAIRWISE])
    ap.add_argument("--force", action="store_true", help="overwrite existing configs")
    args = ap.parse_args()
    base, stem = MODELS[args.model]["base"], MODELS[args.model]["stem"]
    parts = experts(args.model, args.seed)
    out_models = f"../models/task_merge/seed_{args.seed}"
    for method in args.methods:
        merged = f"{stem}_task_merge_4_{method}"
        text = tree_config(method, parts, merged) if method in PAIRWISE else nway_config(method, parts, base)
        path = Path("task_merge", f"seed_{args.seed}", f"{args.model}_task_merge_4_{method}_config.yaml")
        if path.exists() and not args.force:
            print(f"# {path} exists, left as it is (--force to overwrite)", file=sys.stderr)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# Task merge of the four MLP-only {args.model} experts, seed {args.seed}: "
                            "crime, wine, CLINC150, XSum\n# (see make_task_merge_configs.py).\n" + text)
            print(f"# wrote {path}", file=sys.stderr)
        print(method, path, f"{out_models}/{merged}")


if __name__ == "__main__":
    main()
