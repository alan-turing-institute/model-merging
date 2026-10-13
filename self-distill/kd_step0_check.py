"""Step-0 check of the self-distillation loss.

With the untrained student identical to the teacher, the per-token divergence
between them should be ~0 - main's own fix comment records 0.02 nats at the
corrected alignment - yet the training logs report step-0 losses of 7.17 (LoRA)
and inf (full fine-tune). This measures what is actually going on, on the
tokenised eval rows axolotl trained on (its prepared dataset):

  1. KL(teacher || student) per token from the model's own logits, at the
     position the data puts each teacher row (shift 0) and one either side;
  2. the same from logits built the way the KD trainer builds them - last
     hidden state @ lm_head - and how far those are from the model's logits;
  3. cross-entropy per target token, both ways;
  4. axolotl's own fused loss on the same rows, per token (= what eval_loss logs).

Correct alignment and correct logits: (2) matches (1), (4) ~ half of (3), and
(1) is lowest at shift 0. (1) is ~0 only if the student reads the TEACHER's
prompt, so the check also re-scores each row in the teacher's context:

  5. KL and cross-entropy with the student given the teacher's own prompt
     (strict system message and 20-25-word ask) instead of the short one.

If (5) is ~0 the pipeline is sound, and the step-0 gap in the student's context
is the prompt difference self-distillation is meant to close.

    python kd_step0_check.py <prepared_dir> [--rows 32] [--load-in-4bit]
"""

import argparse
import glob

import torch
from datasets import load_from_disk
import importlib.util
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_spec = importlib.util.spec_from_file_location("prep", Path(__file__).with_name("prepare_data_XSum.py"))
prep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(prep)


def teacher_context(tokenizer, input_ids, labels):
    """(teacher-context ids, offset): the same assistant tokens behind the
    teacher's prompt, rebuilt from the article inside the student's prompt."""
    a0 = next(i for i, l in enumerate(labels) if l != -100)
    student_prompt = tokenizer.decode(input_ids[:a0])
    marker = prep.STUDENT_SHORT + "\n\n"
    document = student_prompt.split(marker, 1)[1].split("<end_of_turn>", 1)[0]
    messages = prep.convert_to_prompt({"document": document})["teacher_messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return prompt_ids + list(input_ids[a0:]), len(prompt_ids) - a0


def lm_head_of(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "language_model") and hasattr(base.language_model, "lm_head"):
        return base.language_model.lm_head
    return base.lm_head


def kl_topk(logits, tid, tlp, tmask, shift):
    """Mean per-token KL(teacher || student) over valid teacher rows, with the
    student renormalised over the teacher's top-k, as the fused kernel does."""
    total, n = 0.0, 0
    T = logits.shape[0]
    for i in range(T):
        m = tmask[i].bool()
        j = i + shift
        if not m.any() or j < 0 or j >= T:
            continue
        s = torch.log_softmax(logits[j], -1)[tid[i]]
        s = s - torch.logsumexp(s, -1)
        t = tlp[i]
        total += (t[m].exp() * (t[m] - s[m])).sum().item()
        n += 1
    return total, n


def ce(logits, labels):
    """Mean cross-entropy on target tokens, standard causal shift."""
    lab = labels[1:]
    keep = lab != -100
    lp = torch.log_softmax(logits[:-1][keep], -1)
    return -lp.gather(-1, lab[keep][:, None]).sum().item(), int(keep.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("prepared_dir")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    a = p.parse_args()

    # The eval split is the smaller of axolotl's prepared datasets with KD columns.
    candidates = []
    for d in glob.glob(f"{a.prepared_dir}/*/"):
        try:
            ds = load_from_disk(d)
        except Exception:
            continue
        if "target_logprobs" in ds.column_names:
            candidates.append(ds)
    ds = min(candidates, key=len)
    print(f"eval rows available: {len(ds)}; using {min(a.rows, len(ds))}; columns: {ds.column_names}")

    if a.load_in_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    elif a.load_in_8bit:
        quant = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant = None
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager",
                                                 quantization_config=quant, device_map="cuda").eval()
    print(f"model class: {type(model).__name__}; weights: {'4-bit nf4' if a.load_in_4bit else '8-bit' if a.load_in_8bit else 'bf16'}")
    head = lm_head_of(model)
    tokenizer = AutoTokenizer.from_pretrained(a.model)

    from axolotl.integrations.kd.kernels.liger import LigerFusedLinearKLTopKLogprobLoss
    fused = LigerFusedLinearKLTopKLogprobLoss(0.5, 0.5, 1.0, 0.0, compute_ce_loss=True, normalize_topk=True)

    acc = {k: [0.0, 0] for k in ("kl-1", "kl0", "kl+1", "kl0_trainer", "ce", "ce_trainer", "fused",
                                 "kl_teacher_ctx", "ce_teacher_ctx")}
    max_diff, agree, positions = 0.0, 0, 0
    for row in ds.select(range(min(a.rows, len(ds)))):
        ids = torch.tensor(row["input_ids"], device="cuda")
        labels = torch.tensor(row["labels"], device="cuda")
        tid = torch.tensor(row["target_token_ids"], device="cuda").long()
        tlp = torch.tensor(row["target_logprobs"], device="cuda").float()
        tmask = torch.tensor(row["target_mask"], device="cuda")
        with torch.no_grad():
            out = model(input_ids=ids[None], output_hidden_states=True, return_dict=True)
        L = out.logits[0].float()
        H = out.hidden_states[-1][0]
        F = (H.to(head.weight.dtype) @ head.weight.T).float()
        max_diff = max(max_diff, (L - F).abs().max().item())
        agree += (L.argmax(-1) == F.argmax(-1)).sum().item(); positions += L.shape[0]
        for name, shift in (("kl-1", -1), ("kl0", 0), ("kl+1", 1)):
            t, n = kl_topk(L, tid, tlp, tmask, shift); acc[name][0] += t; acc[name][1] += n
        t, n = kl_topk(F, tid, tlp, tmask, 0); acc["kl0_trainer"][0] += t; acc["kl0_trainer"][1] += n
        t, n = ce(L, labels); acc["ce"][0] += t; acc["ce"][1] += n
        t, n = ce(F, labels); acc["ce_trainer"][0] += t; acc["ce_trainer"][1] += n
        # (5) the same rows behind the teacher's own prompt.
        t_ids, delta = teacher_context(tokenizer, row["input_ids"], row["labels"])
        with torch.no_grad():
            Lt = model(input_ids=torch.tensor(t_ids, device="cuda")[None]).logits[0].float()
        T = Lt.shape[0]
        for i in range(len(row["input_ids"])):
            m = tmask[i].bool()
            if m.any() and 0 <= i + delta < T:
                s = torch.log_softmax(Lt[i + delta], -1)[tid[i]]
                s = s - torch.logsumexp(s, -1)
                acc["kl_teacher_ctx"][0] += (tlp[i][m].exp() * (tlp[i][m] - s[m])).sum().item()
                acc["kl_teacher_ctx"][1] += 1
        t_labels = torch.tensor([-100] * (delta + next(i for i, l in enumerate(row["labels"]) if l != -100))
                                + [l for l in row["labels"][next(i for i, l in enumerate(row["labels"]) if l != -100):]],
                                device="cuda")
        c, n = ce(Lt, t_labels); acc["ce_teacher_ctx"][0] += c; acc["ce_teacher_ctx"][1] += n
        try:
            Hb = H[None].detach().to(head.weight.dtype).requires_grad_(True)
            loss = fused(head.weight, Hb, tid[None], tlp[None], tmask[None], true_labels=labels[None])
            acc["fused"][0] += loss.item(); acc["fused"][1] += int((labels != -100).sum())
        except Exception as e:
            acc["fused"][1] = -1; fused_error = f"{type(e).__name__}: {str(e)[:200]}"

    mean = lambda k: acc[k][0] / acc[k][1] if acc[k][1] > 0 else float("nan")
    print(f"\nmodel logits vs trainer-path logits (last hidden @ lm_head): max |diff| {max_diff:.3f}, "
          f"top-1 agreement {agree / positions:.1%}")
    print(f"KL(teacher||student) per token, model logits:   shift -1 {mean('kl-1'):.4f} | shift 0 {mean('kl0'):.4f} | shift +1 {mean('kl+1'):.4f}")
    print(f"KL(teacher||student) per token, trainer logits: shift 0 {mean('kl0_trainer'):.4f}")
    print(f"cross-entropy per target token: model logits {mean('ce'):.4f} | trainer logits {mean('ce_trainer'):.4f}")
    print(f"0.5*KL + 0.5*CE, trainer logits: {0.5 * mean('kl0_trainer') + 0.5 * mean('ce_trainer'):.4f}")
    print(f"TEACHER'S CONTEXT - KL per token {mean('kl_teacher_ctx'):.4f} | cross-entropy per token {mean('ce_teacher_ctx'):.4f}")
    print("axolotl fused loss per token:", f"{mean('fused'):.4f}" if acc["fused"][1] > 0 else f"not run ({fused_error})")


if __name__ == "__main__":
    main()
