# Writes the task-MoE config of one model and seed: the four MLP-only task
# experts (crime, wine, CLINC150, XSum) as a sparse Mixtral MoE on the untrained
# base, to
#   merge/moe/<model>_task_moe_4_seed_<S>_config.yaml
# and prints "<config> <output dir>" for moe.sbatch.
#
# Usage, from merge/:
#   python make_moe_configs.py <mistral|llama> --seed S [--force]
#
# The settings are those of moe/mistral_task_moe_4_seed_1_config.yaml, whose
# comments explain them: gate_mode hidden, one expert per token, bf16, and as
# each expert's positive prompts its task's training instruction and one real
# training input (routing is per token, so content tokens need something to
# match). The LoRA experts are their convert_to_full_model.sbatch outputs
# (*_mlp_full), since mergekit-moe takes full models; XSum is already full.
# Gemma has no mergekit-moe path, so it is not offered.
import argparse
import sys
from pathlib import Path

import yaml

MODELS = {
    "mistral": {"base": "mistralai/Mistral-7B-Instruct-v0.3", "stem": "mistral_7b"},
    "llama": {"base": "meta-llama/Llama-3.1-8B-Instruct", "stem": "llama3_8b"},
}
# (expert dir under ../models/<task>/seed_<S>/, positive prompts), in expert order.
EXPERTS = {
    "crime": ("{stem}_crime_mlp_full", [
        "Is the following Reddit post about crime? Answer with exactly one word, either 'crime' or 'not_crime', and nothing else.",
        "Cartel Apologizes For Kidnapping, Killing Americans, Turns Over Its Members",
    ]),
    "wine": ("{stem}_wine_mlp_full", [
        "The following is a wine review from Wine Enthusiast magazine. Wines are scored on a scale from 80 to 100 points. Based on the review, predict the wine's points rating. Answer with exactly one whole number between 80 and 100, and nothing else.",
        "Bloody meats, dried cherries, pressed purple flowers, vanilla and blackberry jam arise",
    ]),
    "clinc150": ("{stem}_clinc150_labels_mlp_full", [
        "Classify the user query into exactly one intent label. Answer with the label only.",
        "Labels: accept_reservations, account_blocked, alarm, application_status, apr, are_you_a_bot, balance, bill_balance, bill_due, book_flight, book_hotel",
        "Query: how much is in my checking account",
    ]),
    "xsum": ("{stem}_xsum_gold_long_mlp", [
        "Summarise the following BBC News article in a single sentence of about 20–25 words, in the style of a news headline summary: present tense, third person, no preamble.",
        "Article: In Wales, councils are responsible for funding and overseeing schools. But in England, Mr Osborne's plan will mean local authorities will cease",
    ]),
}


def config(model, seed):
    base, stem = MODELS[model]["base"], MODELS[model]["stem"]
    return {
        "base_model": base,
        "gate_mode": "hidden",
        "experts_per_token": 1,
        "dtype": "bfloat16",
        "experts": [
            {"source_model": f"../models/{task}/seed_{seed}/{name.format(stem=stem)}", "positive_prompts": prompts}
            for task, (name, prompts) in EXPERTS.items()
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--seed", type=int, required=True, help="the seed the experts were trained with")
    ap.add_argument("--force", action="store_true", help="overwrite an existing config")
    args = ap.parse_args()
    path = Path("moe", f"{args.model}_task_moe_4_seed_{args.seed}_config.yaml")
    if path.exists() and not args.force:
        print(f"# {path} exists, left as it is (--force to overwrite)", file=sys.stderr)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Sparse MoE of the four MLP-only {args.model} task experts, seed {args.seed}: crime, wine,\n"
            "# CLINC150, XSum (see make_moe_configs.py, and moe/mistral_task_moe_4_seed_1_config.yaml\n"
            "# for what each setting does). Run with moe.sbatch (mergekit-moe).\n"
            + yaml.safe_dump(config(args.model, args.seed), sort_keys=False, allow_unicode=True, width=1000))
        print(f"# wrote {path}", file=sys.stderr)
    print(path, f"../models/task_merge/seed_{args.seed}/{MODELS[args.model]['stem']}_task_moe_4")


if __name__ == "__main__":
    main()
