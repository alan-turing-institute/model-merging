"""Find the models under a directory and say what each one is.

For consuming a handover: someone puts trained artifacts on the workspace
share, and this works out what they are rather than requiring you to know in
advance. Emits one TSV line per model:

    <name>\t<kind>\t<path>\t<base_model>

  kind        adapter | full
  base_model  for an adapter, read from adapter_config.json's
              base_model_name_or_path - NOT guessed. An adapter evaluated
              against the wrong base produces plausible numbers that mean
              nothing, and the base is recorded in the artifact precisely so it
              does not have to be inferred.

Checkpoint directories are skipped: they are intermediate states of their
parent, not separate models, and treating one as a model is how an
intermediate checkpoint ends up being evaluated as a final result.

    uv run python discover_models.py ~/cloudfiles/code/Users/jmcinroy/model-merging/models
"""

import argparse
import json
import re
import sys
from pathlib import Path

ADAPTER_MARKER = "adapter_config.json"
FULL_MARKER = "config.json"


def classify(directory):
    """(kind, base_model) for a directory, or (None, None) if it isn't a model."""
    if (directory / ADAPTER_MARKER).is_file():
        config = json.loads((directory / ADAPTER_MARKER).read_text())
        return "adapter", config.get("base_model_name_or_path")
    if (directory / FULL_MARKER).is_file():
        return "full", None
    return None, None


def checkpoints(directory):
    return sorted(
        int(m.group(1))
        for p in directory.glob("checkpoint-*")
        if (m := re.fullmatch(r"checkpoint-(\d+)", p.name))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory containing model directories.")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--verbose", action="store_true",
                        help="Also report checkpoints and size, to stderr.")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    found = 0
    seen = set()
    for depth in range(args.max_depth + 1):
        for directory in sorted(root.glob("/".join(["*"] * depth)) if depth else [root]):
            if not directory.is_dir() or directory in seen:
                continue
            # An intermediate checkpoint is not a model in its own right.
            if re.fullmatch(r"checkpoint-\d+", directory.name):
                continue
            # Skip anything already covered by a model found above it.
            if any(parent in seen for parent in directory.parents):
                continue
            kind, base = classify(directory)
            if kind is None:
                continue
            seen.add(directory)
            found += 1
            print(f"{directory.name}\t{kind}\t{directory}\t{base or '-'}")
            if args.verbose:
                steps = checkpoints(directory)
                print(f"    checkpoints={steps or 'none'}", file=sys.stderr)

    if not found:
        sys.exit(f"No models found under {root} "
                 f"(looked for {ADAPTER_MARKER} / {FULL_MARKER} to depth {args.max_depth})")


if __name__ == "__main__":
    main()
