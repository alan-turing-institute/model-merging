"""Check that a trained adapter was trained at the scale you think it was.

This exists because of a specific failure. Registry versions accumulate across
runs, smoke runs get registered under production names, and a 200-example
adapter is indistinguishable from a full-scale one by name, size or
registration timestamp - all three of which were used to pick artifacts here,
and all three were wrong. The thing that finally settled it was the checkpoint
numbers: a run over 200 examples leaves checkpoint-13 and checkpoint-25, a run
over 8000 leaves checkpoint-500 and checkpoint-1000.

So: derive the step count the dataset implies, read what the adapter actually
recorded, and refuse to proceed when they disagree.

    uv run --project evaluate python check_adapter_scale.py \
      --adapter ../models/gemma3-crime-1-of-2-lora \
      --dataset ../datasets/crime_dataset1/train \
      --epochs 3
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Fail if recorded steps are below this fraction of expected.")
    return parser.parse_args()


def recorded_steps(adapter_dir):
    """Highest global_step the adapter's checkpoints attest to."""
    adapter = Path(adapter_dir)
    checkpoints = sorted(
        (int(m.group(1)), p)
        for p in adapter.glob("checkpoint-*")
        if (m := re.fullmatch(r"checkpoint-(\d+)", p.name))
    )
    if not checkpoints:
        return None, []

    highest, path = checkpoints[-1]
    state_file = path / "trainer_state.json"
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        return state.get("global_step", highest), [c for c, _ in checkpoints]
    # The directory name is the step count when there is no state file.
    return highest, [c for c, _ in checkpoints]


def main():
    args = parse_args()

    examples = len(load_from_disk(args.dataset))
    effective_batch = args.micro_batch_size * args.gradient_accumulation_steps
    expected = math.ceil(examples / effective_batch) * args.epochs

    actual, checkpoints = recorded_steps(args.adapter)

    print(f"Adapter  : {args.adapter}")
    print(f"Dataset  : {args.dataset} ({examples} examples)")
    print(f"Expected : ~{expected:.0f} steps "
          f"({examples} / {effective_batch} x {args.epochs:g} epochs)")
    print(f"Recorded : {actual if actual is not None else 'no checkpoints found'}"
          f"  checkpoints={checkpoints}")

    if actual is None:
        print("\nNo checkpoint-* directories, so the training scale cannot be "
              "confirmed from the artifact. Proceed only if you know its provenance.")
        return 0

    if actual < args.tolerance * expected:
        print(f"\nFAIL: {actual} steps is under {args.tolerance:.0%} of the ~{expected:.0f} "
              f"this dataset implies.")
        print("This adapter was almost certainly trained on a smaller (smoke) dataset.")
        print("Pick a different registry version, or retrain.")
        return 1

    if actual > expected / args.tolerance:
        print(f"\nWARNING: {actual} steps is far ABOVE the ~{expected:.0f} this dataset "
              f"implies.")
        print("The adapter was probably trained on a larger dataset, or for more epochs, "
              "than the one it is being paired with here.")
        return 0

    print("\nOK: the recorded step count is consistent with this dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
