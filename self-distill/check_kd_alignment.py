# Measures where a KD dataset's teacher distributions actually belong, by
# comparing them against the teacher model's own next-token distributions.
#
#   uv run python check_kd_alignment.py ../datasets/xsum_kd_data_short [model]
#
# Only meaningful when the teacher was conditioned on the SAME prompt that is
# stored in messages_combined -- i.e. a symmetric dataset. With an asymmetric
# teacher prompt (context distillation) the distributions legitimately differ
# and no shift will read as zero.
#
# Interpretation: the shift with near-zero KL is where the rows belong. Axolotl
# effectively applies shift 0, so a dataset built for Axolotl's own convention
# reads as correct at shift -1, which is the bug KD_ALIGNMENT_BUG.md describes.
# A dataset built by generate_teacher_logprobs_XSum.py carries a compensating
# extra row and should read as correct at shift 0.
import sys

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

SHIFTS = (-2, -1, 0, 1, 2)
N_ROWS = 8


def main():
    path = sys.argv[1]
    model_id = sys.argv[2] if len(sys.argv) > 2 else "google/gemma-3-4b-it"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )

    ds = load_from_disk(path)
    ds = ds["train"] if "train" in getattr(ds, "keys", lambda: [])() else ds

    totals = {s: [] for s in SHIFTS}
    for i in range(min(N_ROWS, len(ds))):
        row = ds[i]
        msgs = row["messages_combined"]
        prompt = tok.apply_chat_template(
            msgs[:-1], tokenize=True, return_dict=False, add_generation_prompt=True
        )
        full = tok.apply_chat_template(msgs, tokenize=True, return_dict=False)
        n_prompt, n_full = len(prompt), len(full)

        with torch.no_grad():
            logits = model(torch.tensor([full]).cuda()).logits[0].float()
        student_logprobs = torch.log_softmax(logits, dim=-1)

        teacher = torch.tensor(row["teacher_logprobs"]).float()
        ids = torch.tensor(row["target_token_ids"]).long()
        # renormalise the teacher over its own top-k, as the plugin does
        teacher_probs = torch.softmax(teacher, dim=-1)

        for shift in SHIFTS:
            kls = []
            for j in range(teacher.shape[0]):
                pos = n_prompt + j + shift
                if not 0 <= pos < n_full:
                    continue
                sl = student_logprobs[pos][ids[j].cuda()].cpu()
                sl = sl - torch.logsumexp(sl, dim=-1)  # same top-k support
                kls.append(
                    float((teacher_probs[j] * (torch.log(teacher_probs[j] + 1e-12) - sl)).sum())
                )
            if kls:
                totals[shift].append(sum(kls) / len(kls))

    print(f"\nmean per-token KL(teacher || student), teacher row j at index P+j+shift")
    print(f"dataset: {path}\n")
    best = min(SHIFTS, key=lambda s: sum(totals[s]) / len(totals[s]))
    for shift in SHIFTS:
        mean = sum(totals[shift]) / len(totals[shift])
        mark = "  <-- best" if shift == best else ""
        print(f"  shift {shift:+d}:  {mean:9.4f} nats{mark}")
    print()
    if best == 0 and sum(totals[0]) / len(totals[0]) < 0.1:
        print("OK: aligned for Axolotl (which applies shift 0).")
    else:
        print(f"WARNING: best shift is {best:+d}, not 0. Axolotl will train on misaligned")
        print("targets, which shows up as immediate word repetition in generations.")


if __name__ == "__main__":
    main()
