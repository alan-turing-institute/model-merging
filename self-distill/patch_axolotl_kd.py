# Patches Axolotl's KD plugin so normalize_logprobs renormalises in log space.
#
# The shipped normalize_logprobs (axolotl/integrations/kd/utils.py, checked
# against axolotl 0.18.0) renormalises the top-k logprobs via probabilities:
# exp, divide by the sum, log. In fp32 an entry more than ~103 nats below the
# row's maximum underflows to exp() == 0 and comes back as log(0) = -inf. With
# kd_normalize_topk the STUDENT's top-k goes through this function, so whenever
# the teacher's top-k holds a token the student all but rules out, the KD loss
# is inf, the gradient is NaN, and one such position corrupts the weights (this
# is what made gemma3_4b_xsum_gold_long diverge). The fix is the mathematically
# identical, stable form: logprobs - logsumexp(logprobs).
#
# The patch edits the installed package, so `uv sync` (or any reinstall of
# axolotl) undoes it: re-run this script afterwards. It is idempotent, and it
# refuses to touch a file whose code it does not recognise.
#
# Usage, from self-distill/ (patches the axolotl in this project's .venv):
#   uv run python patch_axolotl_kd.py
#   uv run python patch_axolotl_kd.py --check   # report only, change nothing
import argparse
import importlib.util
import sys
from pathlib import Path

ORIGINAL = """\
    # Convert logprobs at T_online to probabilities
    # use log sum exp trick to avoid underflow
    position_logprobs_lse = torch.logsumexp(logprobs, dim=-1, keepdim=True)
    teacher_probs_t_online = torch.exp(logprobs - position_logprobs_lse)

    # Normalize probabilities (sum to 1)
    # This is important if the top-k from server aren't a full distribution
    teacher_probs_t_online_sum = teacher_probs_t_online.sum(dim=-1, keepdim=True)
    teacher_probs_t_online = teacher_probs_t_online / teacher_probs_t_online_sum

    final_logprobs_tensor = torch.log(teacher_probs_t_online)

    return final_logprobs_tensor
"""

PATCHED_RETURN = "return logprobs - torch.logsumexp(logprobs, dim=-1, keepdim=True)"

PATCHED = f"""\
    # PATCHED (patch_axolotl_kd.py): renormalise in log space. Going through
    # probabilities (exp, divide by the sum, log) underflows to log(0) = -inf for
    # entries ~103 nats below the row max in fp32, which makes the KD loss inf
    # and its gradient NaN.
    {PATCHED_RETURN}
"""


def utils_path() -> Path:
    # find_spec locates the package without importing it (axolotl's own import
    # is slow and pulls in torchao).
    spec = importlib.util.find_spec("axolotl")
    if spec is None or spec.origin is None:
        sys.exit("axolotl is not installed in this environment")
    return Path(spec.origin).parent / "integrations" / "kd" / "utils.py"


def verify(path: Path):
    """Load the patched file by path and check the underflow case is finite."""
    import torch

    spec = importlib.util.spec_from_file_location("kd_utils_patched", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logprobs = torch.tensor([[0.0, -1.0, -120.0]])
    result = module.normalize_logprobs(logprobs, topk=3)
    expected = torch.log_softmax(logprobs, dim=-1)
    if not torch.isfinite(result).all() or not torch.allclose(result, expected):
        sys.exit(f"verification FAILED: normalize_logprobs gave {result.tolist()}")
    print(f"verified: normalize_logprobs([0, -1, -120]) = {result.tolist()[0]} (finite)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report only; change nothing.")
    args = parser.parse_args()

    path = utils_path()
    text = path.read_text()
    print(f"axolotl KD utils: {path}")

    if ORIGINAL in text:
        if args.check:
            print("NOT patched: the original normalize_logprobs is in place.")
            sys.exit(1)
        path.with_name(path.name + ".orig").write_text(text)
        path.write_text(text.replace(ORIGINAL, PATCHED, 1))
        print(f"patched (original saved as {path.name}.orig)")
    elif PATCHED_RETURN in text:
        print("already patched: nothing to do.")
    else:
        sys.exit(
            "normalize_logprobs does not match axolotl 0.18.0's code, so it was left "
            "alone. Check whether this axolotl version still needs the fix."
        )
    verify(path)


if __name__ == "__main__":
    main()
