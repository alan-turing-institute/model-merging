# Scores the teacher rows Axolotl prepared against a real Llama-3 model, at the
# pairing the KD loss uses and at the pairing one position earlier. GPU.
#
#   uv run python score_alignment.py [prepared_dir] [model]
#
# A Llama-3 model is not the 70B teacher, but a competent model agrees closely
# with the teacher about what comes next and not at all about what came before.
# So KL(teacher row || model) should be small at the correct pairing and large
# at a wrong one.
#   pairing used by the loss:  row t  vs  logits at t
#   pairing if rows were right: row t  vs  logits at t - 1
import glob
import sys

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

PREPARED = sys.argv[1] if len(sys.argv) > 1 else glob.glob("prepared/*/")[0]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.2-1B-Instruct"
N_EXAMPLES = 32
SHIFTS = (-2, -1, 0, 1)


def main():
    assert torch.cuda.is_available(), "no GPU: is a CUDA compat lib on LD_LIBRARY_PATH?"
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager")
        .cuda()
        .eval()
    )
    ds = load_from_disk(PREPARED)

    kl = {s: [] for s in SHIFTS}
    agree = {s: [] for s in SHIFTS}
    for i in range(min(N_EXAMPLES, len(ds))):
        ex = ds[i]
        input_ids = torch.tensor(ex["input_ids"]).cuda()
        mask = torch.tensor(ex["target_mask"])[:, 0].bool().cuda()
        ids = torch.tensor(ex["target_token_ids"]).long().cuda()
        teacher_lp = torch.tensor(ex["target_logprobs"]).float().cuda()
        n = input_ids.shape[0]

        with torch.no_grad():
            logits = model(input_ids[None]).logits[0].float()
        student_lp = torch.log_softmax(logits, dim=-1)

        rows = mask.nonzero().squeeze(-1)
        for s in SHIFTS:
            pos = rows + s
            ok = (pos >= 0) & (pos < n)
            r, p = rows[ok], pos[ok]
            # compare on the teacher's top-k support, both renormalised over it
            t_lp = teacher_lp[r] - torch.logsumexp(teacher_lp[r], -1, keepdim=True)
            s_lp = student_lp[p].gather(-1, ids[r])
            s_lp = s_lp - torch.logsumexp(s_lp, -1, keepdim=True)
            kl[s].append((t_lp.exp() * (t_lp - s_lp)).sum(-1).mean().item())
            t_top1 = ids[r].gather(-1, teacher_lp[r].argmax(-1, keepdim=True)).squeeze(-1)
            agree[s].append((student_lp[p].argmax(-1) == t_top1).float().mean().item())

    print(f"\nmodel: {MODEL}   examples: {len(kl[0])}")
    print("teacher row t scored against the model's prediction at index t + shift\n")
    print(f"  {'shift':>5}  {'mean KL (nats)':>14}  {'top-1 agreement':>15}")
    for s in SHIFTS:
        note = {0: "  <- what the KD loss trains on", -1: "  <- where the rows belong"}.get(s, "")
        print(f"  {s:+5d}  {sum(kl[s]) / len(kl[s]):14.4f}  "
              f"{100 * sum(agree[s]) / len(agree[s]):14.1f}%{note}")


if __name__ == "__main__":
    main()
