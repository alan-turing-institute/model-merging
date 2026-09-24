"""Concatenate save_to_disk datasets into one, for the KD training set.

The two halves' logprobs are produced by separate teacher passes (each expert
over its own half), so they arrive as two datasets and axolotl wants one.

Columns must match exactly - concatenate_datasets raises otherwise, which is
the right behaviour here: a column present in one half and not the other means
the two passes were not run the same way, and silently dropping it would hide
that.
"""

import argparse

from datasets import concatenate_datasets, load_from_disk


from precompute_logprobs import ALIGNMENT_MARKER  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Dataset directories to concatenate.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Shuffle after concatenating so the halves interleave.")
    args = parser.parse_args()

    parts = [load_from_disk(path) for path in args.inputs]
    for path, part in zip(args.inputs, parts):
        print(f"{path}: {len(part)} examples, columns {sorted(part.column_names)}")

    combined = concatenate_datasets(parts)
    if args.shuffle_seed is not None:
        # Without this the first half of training sees only teacher 1 and the
        # second only teacher 2, which confounds "distilling from two teachers"
        # with a curriculum effect.
        combined = combined.shuffle(seed=args.shuffle_seed)

    combined.save_to_disk(args.output)

    # Carry the alignment marker forward, but only if EVERY input has it. The
    # downstream guard refuses a logprobs directory without it, and a
    # concatenation is exactly as trustworthy as its least trustworthy part: one
    # pre-fix half mixed into the union is still a run trained half on noise.
    markers = [Path(d) / ALIGNMENT_MARKER for d in args.inputs]
    if markers and all(m.is_file() for m in markers):
        (Path(args.output) / ALIGNMENT_MARKER).write_text(
            "propagated by concat_datasets.py from: "
            + ", ".join(str(d) for d in args.inputs) + "\n"
        )
        print(f"Alignment marker propagated ({ALIGNMENT_MARKER})")
    else:
        missing = [str(m.parent) for m in markers if not m.is_file()]
        print(f"NO alignment marker written - these inputs lack one: {missing}")

    print(f"\nWrote {len(combined)} examples to {args.output}")


if __name__ == "__main__":
    main()
