# Inspects the dataset exactly as Axolotl's KD plugin prepared it, and asks
# which token each teacher distribution is actually about. CPU only, no model.
#
#   PYTHONPATH=. uv run axolotl preprocess kd_readme_example.yaml
#   uv run python inspect_prepared.py
#
# The KD loss (kernels/liger.py) pairs target_logprobs[t] with the student
# hidden state at t, and that hidden state predicts input_ids[t + 1] -- the
# cross-entropy labels in the same function are shifted left by one to match.
# So for training to be correct, the teacher distribution stored at t must be a
# distribution over input_ids[t + 1].
#
# The teacher (Llama-3.3-70B) generated the assistant turn itself, so its top-1
# token at each step is usually the token that was emitted. That lets us read
# off which token each row belongs to without running any model:
#   top-1 == input_ids[t + 1]  -> row is aligned with what the loss expects
#   top-1 == input_ids[t]      -> row is one position late (the bug)
import glob
import sys

from datasets import load_from_disk
from transformers import AutoTokenizer

PREPARED = sys.argv[1] if len(sys.argv) > 1 else glob.glob("prepared/*/")[0]
TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"
N_SHOWN = 12


def top1(ids, logprobs):
    return ids[max(range(len(logprobs)), key=logprobs.__getitem__)]


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ds = load_from_disk(PREPARED)
    print(f"prepared dataset: {PREPARED}  ({len(ds)} examples)\n")

    n = same_as_current = same_as_next = 0
    first_row_vs_first_label = {}
    for ex in ds:
        input_ids, labels = ex["input_ids"], ex["labels"]
        mask, ids, lps = ex["target_mask"], ex["target_token_ids"], ex["target_logprobs"]

        # where Axolotl started laying down real teacher rows, and where the
        # assistant turn (the first trained label) starts
        first_row = next(t for t, row in enumerate(lps) if row[0] != float("-inf"))
        first_label = next(t for t, lab in enumerate(labels) if lab != -100)
        d = first_row - first_label
        first_row_vs_first_label[d] = first_row_vs_first_label.get(d, 0) + 1

        for t in range(len(input_ids) - 1):
            if not mask[t][0]:
                continue
            n += 1
            best = top1(ids[t], lps[t])
            same_as_current += best == input_ids[t]
            same_as_next += best == input_ids[t + 1]

    print("first teacher row index - first assistant label index:")
    for d, c in sorted(first_row_vs_first_label.items()):
        print(f"  {d:+d}: {c} examples")
    print()
    print(f"unmasked KD positions t: {n}")
    print(f"  teacher top-1 == input_ids[t]    (the token already consumed): "
          f"{same_as_current:6d}  ({100 * same_as_current / n:5.1f}%)")
    print(f"  teacher top-1 == input_ids[t+1]  (the token the loss targets):  "
          f"{same_as_next:6d}  ({100 * same_as_next / n:5.1f}%)")
    print()

    # one example, position by position, as the loss sees it
    ex = ds[0]
    input_ids, mask = ex["input_ids"], ex["target_mask"]
    ids, lps = ex["target_token_ids"], ex["target_logprobs"]
    start = next(t for t in range(len(input_ids)) if mask[t][0])
    print(f"example 0, first {N_SHOWN} KD positions. At index t the student predicts "
          "input_ids[t+1],\nand the KD loss asks it to match the teacher row stored at t:\n")
    print(f"  {'t':>5}  {'input_ids[t]':>16}  {'student predicts':>16}  {'teacher row top-1':>18}")
    for t in range(start, min(start + N_SHOWN, len(input_ids) - 1)):
        cur = tok.decode([input_ids[t]])
        nxt = tok.decode([input_ids[t + 1]])
        best = tok.decode([top1(ids[t], lps[t])])
        print(f"  {t:5d}  {cur!r:>16}  {nxt!r:>16}  {best!r:>18}")
    print()

    if same_as_current > same_as_next:
        print("BUG: each teacher distribution sits at the index of the token it produced,")
        print("so the student at t is trained to predict input_ids[t] -- the token it has")
        print("just read -- instead of input_ids[t+1]. The KD targets are one position late.")
    else:
        print("OK: teacher distributions are aligned with the student's next-token prediction.")


if __name__ == "__main__":
    main()
