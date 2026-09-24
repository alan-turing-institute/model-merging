# Runs the teacher model over the XSum prompts (from "prepare_data_XSum.py")
# and records its generations plus per-token top-k logprobs, in the format
# expected by Axolotl's KD plugin (axolotl.integrations.kd.chat_template).
#
# The teacher is conditioned on `teacher_messages` (strict system prompt), but
# the rows we store pair its output with a DIFFERENT, plainer student prompt.
# That is context distillation: the student learns to produce strict-prompt
# output from a plain prompt. Two datasets are written from a single generation
# pass, since the logprobs do not depend on the student prompt -- so the two
# training runs differ in exactly one variable.
#
# The KD plugin never receives an explicit offset: it infers where the teacher
# distributions belong from the lengths alone, in
# ChatTemplateStrategyWithKDv2.transform_logprobs:
#
#     input_padding_len = len(input_ids) - len(teacher_logprobs)
#
# where `input_ids` is the *whole* templated conversation. Axolotl's own
# convention is off by one against its loss kernel (see KD_ALIGNMENT_BUG.md), so
# we deliberately emit ONE MORE row than that convention calls for:
#
#     len(teacher_logprobs) == len(template(messages_combined))
#                              - len(template(student_messages, add_generation_prompt=True))
#                              + 1
#
# vLLM gives us one logprob position per generated token (content + the
# <end_of_turn> that stopped it). Gemma's chat template emits a further "\n"
# after <end_of_turn>, so the templated assistant turn is one token longer than
# the generation. We close that gap explicitly with a one-hot row per leftover
# template token; without it every teacher distribution lands on the wrong
# token and training learns noise.
import os
from pathlib import Path

# Isambard has no CUDA toolkit, so FlashInfer cannot JIT-compile its sampling
# kernels ("Could not find nvcc"). vLLM's PyTorch-native sampler needs no
# compiler and changes nothing here: the stored logprobs come from
# raw_logprobs (pre-sampling logits) either way. Set before importing vllm.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

TEACHER_MODEL = "google/gemma-3-1b-it"
TOP_K = 20  # must stay fixed across the whole dataset
TEMPERATURE = 0.0  # greedy: cleaner targets, and the top-k logprobs are T=1 either way
MAX_TOKENS = 128
# Must match `sequence_len` in the Axolotl config. Rows that cannot fit are
# dropped here rather than left for Axolotl to truncate: its truncation path
# mangles the logprobs instead of erroring.
MAX_SEQ_LEN = 4096
# Headroom for the template tokens that wrap the assistant turn.
TEMPLATE_SLACK = 8
# Placeholder mass for the filler entries in a one-hot row. Small enough to be
# irrelevant to the KL, but *not* -inf: the loss computes p * (log p - log q),
# and exp(-inf) * -inf is NaN.
FILLER_LOGPROB = -30.0

# student prompt field -> output dataset name
STUDENT_VARIANTS = {
    "messages_long": "xsum_kd_data_long",
    "messages_short": "xsum_kd_data_short",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
PROMPTS_PATH = DATA_DIR / "xsum_prompts"


def token_ids_for(tokenizer, messages, add_generation_prompt=False):
    """Token ids exactly as Axolotl will produce them.

    return_dict=False mirrors Axolotl's own call and yields a plain id list;
    transformers 5.x otherwise hands back a BatchEncoding.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
    )


def one_hot_row(token_id):
    """A top-k row asserting `token_id` with probability ~1.

    Used for the deterministic template tokens that follow the generation, so
    that the logprob array lines up with the templated conversation.
    """
    fillers = [t for t in range(TOP_K + 1) if t != token_id][: TOP_K - 1]
    return [0.0] + [FILLER_LOGPROB] * (TOP_K - 1), [token_id] + fillers


def build_student_row(tokenizer, student_messages, gen_token_ids, teacher_logprobs,
                      target_token_ids, text):
    """Pair the teacher's output with one student prompt, or None if unusable."""
    prompt_ids = token_ids_for(tokenizer, student_messages, add_generation_prompt=True)
    messages_combined = list(student_messages) + [{"role": "assistant", "content": text}]
    full_ids = token_ids_for(tokenizer, messages_combined)

    n_prompt = len(prompt_ids)
    # Re-templating must reproduce both the prompt and the tokens the teacher
    # generated. Gemma's template trims assistant content, and BPE can re-merge
    # across the turn boundary, so verify rather than assume.
    if (
        full_ids[:n_prompt] != list(prompt_ids)
        or full_ids[n_prompt : n_prompt + len(gen_token_ids)] != gen_token_ids
        or len(full_ids) > MAX_SEQ_LEN
    ):
        return None

    logprobs = list(teacher_logprobs)
    token_ids = list(target_token_ids)
    # Close the gap to the templated assistant turn (for Gemma: the "\n" after
    # <end_of_turn>).
    for token_id in full_ids[n_prompt + len(gen_token_ids) :]:
        logprob_row, token_id_row = one_hot_row(token_id)
        logprobs.append(logprob_row)
        token_ids.append(token_id_row)

    # OFF-BY-ONE CORRECTION -- see KD_ALIGNMENT_BUG.md.
    # Axolotl places teacher row j at index (len(input_ids) - len(logprobs)) + j,
    # but a causal LM's hidden state at index i predicts token i+1. Emitting one
    # extra row makes input_padding_len one smaller, so row j lands at P+j-1,
    # which is the position whose prediction IS token P+j.
    # Measured KL(teacher || student) with teacher == student: 0.02 nats at this
    # shift, against 17.6 nats at Axolotl's default. Without this the KD term
    # trains the model to predict the token it just emitted, which showed up as
    # immediate word repetition in 35% of generations.
    # The extra row lands at the final index, whose "next token" does not exist;
    # Axolotl masks it (labels[-1] == -100), so its contents are irrelevant.
    filler_logprobs, filler_token_ids = one_hot_row(full_ids[-1])
    logprobs.append(filler_logprobs)
    token_ids.append(filler_token_ids)

    if len(logprobs) != len(full_ids) - n_prompt + 1:
        return None

    return {
        "messages_combined": messages_combined,
        "teacher_logprobs": logprobs,
        "target_token_ids": token_ids,
    }


def generate_split(llm, tokenizer, sampling_params, split, split_name):
    prompts = []
    kept = []
    dropped = {"too_long": 0, "truncated": 0, "retokenised": 0, "short_topk": 0}

    for example in split:
        teacher_ids = token_ids_for(
            tokenizer, example["teacher_messages"], add_generation_prompt=True
        )
        longest_student = max(
            len(token_ids_for(tokenizer, example[f], add_generation_prompt=True))
            for f in STUDENT_VARIANTS
        )
        if (
            max(len(teacher_ids), longest_student) + MAX_TOKENS + TEMPLATE_SLACK
            > MAX_SEQ_LEN
        ):
            dropped["too_long"] += 1
            continue
        # Pass ids, not text: vLLM re-tokenises string prompts with
        # add_special_tokens=True, which would prepend a second <bos> to a
        # template that already emits one.
        prompts.append(TokensPrompt(prompt_token_ids=teacher_ids))
        kept.append(example)

    outputs = llm.generate(prompts, sampling_params)

    rows = {name: [] for name in STUDENT_VARIANTS.values()}
    for example, output in zip(kept, outputs):
        gen = output.outputs[0]

        # A generation cut off by max_tokens has no <end_of_turn>, so the
        # length relationship the KD plugin relies on no longer holds.
        if gen.finish_reason != "stop" or not gen.logprobs:
            dropped["truncated"] += 1
            continue

        gen_token_ids = list(gen.token_ids)
        text = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()

        teacher_logprobs = []
        target_token_ids = []
        for position_logprobs in gen.logprobs:
            # vLLM inserts the sampled token first and the top-k after it, so
            # the dict is not in rank order and can hold TOP_K + 1 entries.
            entries = sorted(
                position_logprobs.items(), key=lambda kv: kv[1].logprob, reverse=True
            )[:TOP_K]
            teacher_logprobs.append([logprob.logprob for _, logprob in entries])
            target_token_ids.append([token_id for token_id, _ in entries])

        # The plugin derives one top-k width for the row from the *least
        # frequent* row length, so a single short row collapses the whole
        # sample to that width. Drop the sample instead.
        if any(len(row) != TOP_K for row in teacher_logprobs):
            dropped["short_topk"] += 1
            continue

        # A row is kept only if it validates for EVERY student variant, so the
        # two datasets stay row-for-row comparable.
        built = {
            name: build_student_row(
                tokenizer, example[field], gen_token_ids,
                teacher_logprobs, target_token_ids, text,
            )
            for field, name in STUDENT_VARIANTS.items()
        }
        if any(r is None for r in built.values()):
            dropped["retokenised"] += 1
            continue
        for name, row in built.items():
            rows[name].append(row)

    n_kept = len(next(iter(rows.values())))
    print(f"[{split_name}] kept {n_kept}/{len(split)} (dropped: {dropped})")
    return rows


def main():
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    # enforce_eager skips torch.compile and CUDA-graph capture, which did not
    # complete within 15 minutes here on a cold cache. Eager init takes ~6s and
    # costs only some decode throughput.
    llm = LLM(
        model=TEACHER_MODEL,
        max_model_len=MAX_SEQ_LEN,
        seed=42,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        logprobs=TOP_K,
    )

    prompts_ds = load_from_disk(str(PROMPTS_PATH))

    per_dataset = {name: {} for name in STUDENT_VARIANTS.values()}
    for split in prompts_ds:
        rows = generate_split(
            llm, tokenizer, sampling_params, prompts_ds[split], split
        )
        for name, split_rows in rows.items():
            per_dataset[name][split] = Dataset.from_list(split_rows)

    for name, splits in per_dataset.items():
        out = DATA_DIR / name
        DatasetDict(splits).save_to_disk(str(out))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
