# Writes the task-MoE config of one model and seed: the four MLP-only task
# experts (crime, wine, CLINC150, XSum) as a sparse Mixtral MoE on the untrained
# base, to
#   merge/moe/<model>_task_moe_4_seed_<S>_config.yaml
# and prints "<config> <output dir>" for moe.sbatch.
#
# Usage, from merge/:
#   python make_moe_configs.py <mistral|llama> --seed S [--gate-mode hidden|random|prompt_mean] [--force]
#
# One expert per token, bf16. The LoRA experts are their
# convert_to_full_model.sbatch outputs (*_mlp_full), since mergekit-moe takes
# full models; XSum is already full. Gemma has no mergekit-moe path, so it is
# not offered. Three routers (--gate-mode), each its own model:
#   hidden       <stem>_task_moe_4: mergekit's hidden gate with the positive
#                prompts below. mergekit uses only the LAST token of each prompt
#                (raw text, no chat template), from the residual stream entering
#                each layer, not the router's own input: see the comments in
#                moe/mistral_task_moe_4_seed_1_config.yaml. The prompts are kept
#                as they were run, slips included (XSum's old 20-25 words; 11 of
#                CLINC150's 150 labels).
#   random       <stem>_task_moe_4_random: random routers, the control.
#   prompt_mean  <stem>_task_moe_4_prompt_mean: mergekit builds the MoE with a
#                random router, which set_moe_router.py then replaces with one
#                built from the router's own inputs on real example prompts in
#                the evaluation's format (moe/prompts_<model>.json, from
#                make_moe_prompts.py; XSum's current 18-24 words prompt, all 150
#                CLINC150 labels). moe.sbatch does both (--router-prompts).
import argparse
import sys
from pathlib import Path

import yaml

MODELS = {
    "mistral": {"base": "mistralai/Mistral-7B-Instruct-v0.3", "stem": "mistral_7b"},
    "llama": {"base": "meta-llama/Llama-3.1-8B-Instruct", "stem": "llama3_8b"},
}
# (expert dir under ../models/<task>/seed_<S>/, positive prompts), in expert order.
# The hidden gate's positive prompts are kept exactly as the <stem>_task_moe_4
# models were built with them, slips included (see the header), so that those
# configs can be regenerated; prompt_mean's prompts are in moe/prompts_<model>.json.
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


def config(model, seed, gate_mode="hidden"):
    base, stem = MODELS[model]["base"], MODELS[model]["stem"]
    # gate_mode random: each layer's router is a random projection of the token's
    # hidden state (torch.randn, drawn under mergekit-moe's --random-seed, which
    # moe.sbatch sets to the seed). Routing still depends on the input but knows
    # nothing about the tasks: the control for the hidden gate. It uses no
    # prompts, so none are written. prompt_mean is built the same way, its
    # router then replaced by set_moe_router.py.
    mergekit_gate = "random" if gate_mode == "prompt_mean" else gate_mode
    def expert(task, name, prompts):
        e = {"source_model": f"../models/{task}/seed_{seed}/{name.format(stem=stem)}"}
        if mergekit_gate != "random":
            e["positive_prompts"] = prompts
        return e
    return {
        "base_model": base,
        "gate_mode": mergekit_gate,
        "experts_per_token": 1,
        "dtype": "bfloat16",
        "experts": [expert(task, name, prompts) for task, (name, prompts) in EXPERTS.items()],
    }


def names(model, seed, gate_mode):
    """The config path and the output dir: <stem>_task_moe_4[_random]."""
    tag = "" if gate_mode == "hidden" else f"_{gate_mode}"
    return (Path("moe", f"{model}_task_moe_4{tag}_seed_{seed}_config.yaml"),
            f"../models/task_merge/seed_{seed}/{MODELS[model]['stem']}_task_moe_4{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--seed", type=int, required=True, help="the seed the experts were trained with")
    ap.add_argument("--gate-mode", choices=["hidden", "random", "prompt_mean"], default="hidden",
                    help="hidden: mergekit's gate from task prompts; random: the control; "
                         "prompt_mean: set_moe_router.py's gate from example prompts")
    ap.add_argument("--force", action="store_true", help="overwrite an existing config")
    args = ap.parse_args()
    path, out = names(args.model, args.seed, args.gate_mode)
    if path.exists() and not args.force:
        print(f"# {path} exists, left as it is (--force to overwrite)", file=sys.stderr)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        what = {"random": "# Randomly routed (gate_mode random): the control for the prompt-routed MoE.\n",
                "prompt_mean": "# Built with a random router, which moe.sbatch --router-prompts then replaces\n"
                               f"# (set_moe_router.py, moe/prompts_{args.model}.json): gate_mode prompt_mean.\n",
                }.get(args.gate_mode, "")
        path.write_text(
            f"# Sparse MoE of the four MLP-only {args.model} task experts, seed {args.seed}: crime, wine,\n"
            "# CLINC150, XSum (see make_moe_configs.py, and moe/mistral_task_moe_4_seed_1_config.yaml\n"
            "# for what each setting does). Run with moe.sbatch (mergekit-moe).\n"
            + what
            + yaml.safe_dump(config(args.model, args.seed, args.gate_mode), sort_keys=False,
                             allow_unicode=True, width=1000))
        print(f"# wrote {path}", file=sys.stderr)
    print(path, out)


if __name__ == "__main__":
    main()
