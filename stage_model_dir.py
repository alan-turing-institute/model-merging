"""Stage a model directory, filling in processor files it is missing.

mergekit writes weights and tokenizer files, but not the image-processor
configs a multimodal base model ships. gemma-3-4b-it is multimodal, so axolotl
loads a processor alongside the tokenizer and a merged gemma model fails with:

    OSError: Can't load image processor for '<dir>' ... containing a
    preprocessor_config.json file

even though nothing here touches images. Rather than copy tens of gigabytes to
add a few kilobytes of JSON, this stages a directory of SYMLINKS to the source
and fills only the missing files from the base model.

It also makes a read-only handover usable: a colleague's share directory cannot
be written into, and this needs no write access to it.

    uv run --project evaluate python stage_model_dir.py \
      --source /shared/gemma3-crime-merged-linear-2 \
      --dest models/gemma3-crime-merged-linear-2 \
      --base-model google/gemma-3-4b-it
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

# Files a processor needs that a merge may omit. Absent-upstream is fine - not
# every base model has every one - so a failed download is skipped, not fatal.
PROCESSOR_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--base-model", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    dest = Path(args.dest)

    if not source.is_dir():
        raise SystemExit(f"Source is not a directory: {source}")

    # A prior run may have left a symlink here rather than a directory.
    if dest.is_symlink():
        dest.unlink()
    dest.mkdir(parents=True, exist_ok=True)

    linked = 0
    for entry in sorted(source.iterdir()):
        target = dest / entry.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(entry)
        linked += 1
    print(f"Staged {dest} -> {source} ({linked} entries linked)")

    added, missing = [], []
    for name in PROCESSOR_FILES:
        if (dest / name).exists():
            continue
        try:
            path = hf_hub_download(repo_id=args.base_model, filename=name)
        except Exception as exc:  # noqa: BLE001 - absent upstream is expected
            missing.append(f"{name} ({type(exc).__name__})")
            continue
        # A real copy, not a link: the HF cache is not a stable location to
        # point a model directory at.
        shutil.copy2(path, dest / name)
        added.append(name)

    print(f"Added from {args.base_model}: {added or 'nothing needed'}")
    if missing:
        print(f"Not available upstream (fine if the base has no processor): {missing}")


if __name__ == "__main__":
    main()
