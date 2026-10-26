# Gives a trained or merged model its source model's end-of-turn stop tokens.
#
# Gemma 3 ends a reply with <end_of_turn> (106), so google/gemma-3-*-it set
# eos_token_id [1, 106] in both config.json and generation_config.json. Two
# things lose the 106:
#   - Axolotl's model loader replaces config.eos_token_id with the tokenizer's
#     single eos id (1) whenever they differ, which a list always does. Trained
#     models still save generation_config.json with [1, 106], so they stop.
#   - mergekit writes no generation_config.json and copies config.json from the
#     base model (or the first model when there is none). A merge whose
#     config.json came from a trained model therefore has eos 1 alone, never
#     stops on <end_of_turn>, and generates to max_tokens on every sample: XSum
#     evaluations of such merges took 1 h instead of 10 min (23 Oct).
#
# For <model_dir> this copies the source's generation_config.json if the dir has
# none, and sets config.json's eos_token_id to the source's. The source is the
# config.yaml base_model for an Axolotl output, or for a mergekit output its
# base_model, else its first model (the base half of "<base>+<adapter>"). A
# local source without a generation_config.json is followed to its own source.
#
# Usage: python fix_eos.py <model_dir> [--base-dir DIR]
# Relative paths in the config resolve against --base-dir (default: the current
# directory, i.e. the dir the job ran from), as for provenance.py.
import argparse
import json
import os
import shutil
from pathlib import Path

GENERATION_CONFIG = "generation_config.json"


def resolve(path, base_dir):
    """path made absolute against base_dir, textually (models/ is a symlink)."""
    return os.path.normpath(os.path.join(os.path.abspath(base_dir), path))


def source_of(model_dir, base_dir):
    """The model a trained or merged dir was made from: a local dir or a Hub id."""
    import yaml
    merge_config = Path(model_dir, "mergekit_config.yml")
    if merge_config.exists():
        cfg = yaml.safe_load(merge_config.read_text())
        spec = cfg.get("base_model") or cfg["models"][0]["model"]
        spec = spec.partition("+")[0]
    else:
        spec = yaml.safe_load(Path(model_dir, "config.yaml").read_text())["base_model"]
    local = resolve(spec, base_dir)
    return local if os.path.isdir(local) else spec


def source_generation_config(model_dir, base_dir):
    """Path to the generation_config.json model_dir should have."""
    source = source_of(model_dir, base_dir)
    if not os.path.isdir(source):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(source, GENERATION_CONFIG)
    path = Path(source, GENERATION_CONFIG)
    if path.exists():
        return str(path)
    # One of our own models without one (e.g. an older merge): its paths
    # resolve against the dir its job ran from, which mergekit doesn't record;
    # merge/ is the only place merges run from.
    return source_generation_config(source, Path(__file__).resolve().parent / "merge")


def fix(model_dir, base_dir="."):
    """Bring model_dir's stop tokens in line with its source's; print what changed."""
    if not Path(model_dir, "config.json").exists():
        # A LoRA adapter: it loads with its base model's configs, which are intact.
        print(f"{model_dir}: no config.json (an adapter?), nothing to fix")
        return
    src = source_generation_config(model_dir, base_dir)
    eos = json.loads(Path(src).read_text())["eos_token_id"]
    dst = Path(model_dir, GENERATION_CONFIG)
    if not dst.exists():
        shutil.copyfile(src, dst)
        print(f"copied {src} -> {dst}")
    config_path = Path(model_dir, "config.json")
    config = json.loads(config_path.read_text())
    if config.get("eos_token_id") != eos:
        print(f"{config_path}: eos_token_id {config.get('eos_token_id')} -> {eos}")
        config["eos_token_id"] = eos
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Restore a model's end-of-turn stop tokens.")
    parser.add_argument("model_dir", help="An Axolotl or mergekit output directory.")
    parser.add_argument(
        "--base-dir", default=".",
        help="Directory relative paths in the config resolve against (default: the current one).",
    )
    args = parser.parse_args()
    fix(args.model_dir, args.base_dir)


if __name__ == "__main__":
    main()
