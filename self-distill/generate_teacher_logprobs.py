# Runs a teacher model over a prompts dataset (e.g. from "prepare_data_XSum.py")
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
# convention is off by one against its loss kernel (see build_student_row), so
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
#
# With --targets gold the assistant turn is the dataset's reference answer (the
# `reference` column the prepare_data_*.py scripts emit) instead of a
# generation, and the teacher only scores it (vLLM prompt_logprobs over teacher
# prompt + gold reference). The KD plugin computes CE and KD on the
# same sequence, so this is how CE trains on the gold text while KD still
# carries the teacher's distribution at each of its tokens.
#
# Each student field gets its own output dataset,
# <output_dir>/<prefix>_<field without "messages_">, e.g. by default
# <output_dir>/kd_data_long and <output_dir>/kd_data_short. Each also gets a
# provenance.json (see ../provenance.py) recording the teacher, targets, source
# prompts and their own record, the arguments, git commit and Slurm job: unlike
# training, generation has no config file to say how a dataset was built. Its
# coverage is the prompts dataset's ("full", or e.g. "1 of 2" for a piece).
#
# Usage:
#   python generate_teacher_logprobs.py google/gemma-3-4b-it \
#       ../datasets/xsum/xsum_prompts ../datasets/xsum/gemma3_4b_xsum_kd_data
#   python generate_teacher_logprobs.py path/to/teacher \
#       ../datasets/xsum/split_4/xsum_prompts_1_of_4 ../datasets/xsum/split_4/gemma3_4b_xsum_kd_data_1_of_4 --top-k 50
import argparse
import os
import sys
from pathlib import Path

# Isambard has no CUDA toolkit, so FlashInfer cannot JIT-compile its sampling
# kernels ("Could not find nvcc"). vLLM's PyTorch-native sampler needs no
# compiler and changes nothing here: the stored logprobs come from
# raw_logprobs (pre-sampling logits) either way. Set before importing vllm.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams, TokensPrompt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import provenance  # noqa: E402  (repo root, shared with split_dataset.py etc.)

# Defaults for the command-line options; see parse_args for what each one does.
TOP_K = 20
TEMPERATURE = 0.0
MAX_TOKENS = 128
MAX_SEQ_LEN = 4096
SEED = 42
# Headroom for the template tokens that wrap the assistant turn.
TEMPLATE_SLACK = 8
# Placeholder mass for the filler entries in a one-hot row. Small enough to be
# irrelevant to the KL, but *not* -inf: the loss computes p * (log p - log q),
# and exp(-inf) * -inf is NaN.
FILLER_LOGPROB = -30.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate teacher outputs and top-k logprobs for Axolotl's KD plugin."
    )
    parser.add_argument("model", help="Teacher model: a Hugging Face Hub address or a local directory.")
    parser.add_argument("dataset_dir", type=Path, help="Prompts dataset saved with save_to_disk.")
    parser.add_argument("output_dir", type=Path, help="Directory to write the KD datasets into.")
    parser.add_argument(
        "--prefix", default="kd_data",
        help="Name prefix for each output dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--teacher-field", default="teacher_messages",
        help="Column holding the prompt the teacher generates from (default: %(default)s).",
    )
    parser.add_argument(
        "--student-fields", nargs="+", default=["messages_long", "messages_short"],
        help="Columns holding student prompts; one output dataset is written per field "
        "(default: %(default)s).",
    )
    # "gold" keeps the KD plugin's single-sequence layout but makes that sequence
    # the dataset's reference answer: the CE term then trains on the gold text
    # and the KD term on the teacher's distribution at each gold token (teacher
    # forcing via vLLM's prompt_logprobs). --temperature and --max-tokens only
    # affect "teacher" mode, though --max-tokens still bounds the length check.
    parser.add_argument(
        "--targets", choices=["teacher", "gold"], default="teacher",
        help="Assistant turn to train on: the teacher's own generation, or the gold "
        "reference scored by the teacher (default: %(default)s).",
    )
    parser.add_argument(
        "--gold-field", default="reference",
        help="Column holding the gold answer for --targets gold. The prepare_data_*.py "
        "scripts write it as `reference`; prompts datasets made before that have only "
        "XSum's `summary` (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help="Logprobs stored per position; must stay fixed across the whole dataset "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help="Sampling temperature. 0 is greedy, which gives cleaner targets; the stored "
        "top-k logprobs are T=1 either way (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help="Maximum tokens the teacher may generate (default: %(default)s).",
    )
    # Rows that cannot fit are dropped here rather than left for Axolotl to
    # truncate: its truncation path mangles the logprobs instead of erroring.
    parser.add_argument(
        "--max-seq-len", type=int, default=MAX_SEQ_LEN,
        help="Must match `sequence_len` in the Axolotl config (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="vLLM seed (default: %(default)s).")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs to shard the teacher across (default: %(default)s).",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="Fraction of GPU memory vLLM may use (default: %(default)s).",
    )
    # enforce_eager skips torch.compile and CUDA-graph capture, which did not
    # complete within 15 minutes here on a cold cache. Eager init takes ~6s and
    # costs only some decode throughput.
    parser.add_argument(
        "--no-enforce-eager", dest="enforce_eager", action="store_false",
        help="Let vLLM compile and capture CUDA graphs (slow to start on a cold cache).",
    )
    return parser.parse_args()


def output_name(prefix, field):
    return f"{prefix}_{field.removeprefix('messages_')}"


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


def one_hot_row(token_id, top_k):
    """A top-k row asserting `token_id` with probability ~1.

    Used for the deterministic template tokens that follow the generation, so
    that the logprob array lines up with the templated conversation.
    """
    fillers = [t for t in range(top_k + 1) if t != token_id][: top_k - 1]
    return [0.0] + [FILLER_LOGPROB] * (top_k - 1), [token_id] + fillers


def build_student_row(tokenizer, student_messages, gen_token_ids, teacher_logprobs,
                      target_token_ids, text, top_k, max_seq_len):
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
        or len(full_ids) > max_seq_len
    ):
        return None

    logprobs = list(teacher_logprobs)
    token_ids = list(target_token_ids)
    # Close the gap to the templated assistant turn (for Gemma: the "\n" after
    # <end_of_turn>).
    for token_id in full_ids[n_prompt + len(gen_token_ids) :]:
        logprob_row, token_id_row = one_hot_row(token_id, top_k)
        logprobs.append(logprob_row)
        token_ids.append(token_id_row)

    # OFF-BY-ONE CORRECTION.
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
    filler_logprobs, filler_token_ids = one_hot_row(full_ids[-1], top_k)
    logprobs.append(filler_logprobs)
    token_ids.append(filler_token_ids)

    if len(logprobs) != len(full_ids) - n_prompt + 1:
        return None

    return {
        "messages_combined": messages_combined,
        "teacher_logprobs": logprobs,
        "target_token_ids": token_ids,
    }


def stop_token_ids(model, tokenizer):
    """The token ids vLLM stops generation on: generation_config's eos_token_id(s).

    For Gemma 3 that is <eos> and <end_of_turn>. Gold mode cuts the templated
    gold answer at the same place, so gold rows and generated rows share one
    layout for any model family.
    """
    try:
        eos = GenerationConfig.from_pretrained(model).eos_token_id
    except OSError:  # no generation_config.json
        eos = None
    if eos is None:
        eos = tokenizer.eos_token_id
    return set(eos if isinstance(eos, list) else [eos])


def gold_token_ids(tokenizer, teacher_messages, teacher_ids, reference, stop_ids):
    """The gold reference's assistant tokens as the teacher sees them, or None.

    Returned in the same shape as a vLLM generation: the content tokens plus the
    stop token that closes them (for Gemma, <end_of_turn>), without whatever the
    template emits after it (Gemma's trailing "\n", which build_student_row adds
    back as a one-hot row).
    """
    full_ids = token_ids_for(
        tokenizer, list(teacher_messages) + [{"role": "assistant", "content": reference}]
    )
    if full_ids[: len(teacher_ids)] != list(teacher_ids):
        return None
    tail = full_ids[len(teacher_ids):]
    cut = next((i for i, t in enumerate(tail) if t in stop_ids), None)
    if cut is None:
        return None
    return tail[: cut + 1]


def topk_rows(positions, top_k):
    """Split vLLM's per-position {token_id: Logprob} dicts into sorted top-k rows."""
    teacher_logprobs = []
    target_token_ids = []
    for position_logprobs in positions:
        # vLLM inserts the sampled (or, for prompt_logprobs, the actual) token
        # first and the top-k after it, so the dict is not in rank order and can
        # hold top_k + 1 entries.
        entries = sorted(
            position_logprobs.items(), key=lambda kv: kv[1].logprob, reverse=True
        )[:top_k]
        teacher_logprobs.append([logprob.logprob for _, logprob in entries])
        target_token_ids.append([token_id for token_id, _ in entries])
    return teacher_logprobs, target_token_ids


def generate_split(llm, tokenizer, sampling_params, split, split_name, args, stop_ids):
    prompts = []
    kept = []
    dropped = {
        "too_long": 0, "bad_gold": 0, "truncated": 0, "retokenised": 0, "short_topk": 0,
    }

    for example in split:
        teacher_ids = token_ids_for(
            tokenizer, example[args.teacher_field], add_generation_prompt=True
        )
        longest_student = max(
            len(token_ids_for(tokenizer, example[f], add_generation_prompt=True))
            for f in args.student_fields
        )
        if (
            max(len(teacher_ids), longest_student) + args.max_tokens + TEMPLATE_SLACK
            > args.max_seq_len
        ):
            dropped["too_long"] += 1
            continue
        gold_ids = None
        if args.targets == "gold":
            gold_ids = gold_token_ids(
                tokenizer, example[args.teacher_field], teacher_ids,
                example[args.gold_field].strip(), stop_ids,
            )
            if gold_ids is None or len(gold_ids) > args.max_tokens:
                dropped["bad_gold"] += 1
                continue
        # Pass ids, not text: vLLM re-tokenises string prompts with
        # add_special_tokens=True, which would prepend a second <bos> to a
        # template that already emits one. In gold mode the prompt carries the
        # gold reference too, so prompt_logprobs scores it under the teacher.
        prompt_ids = list(teacher_ids) + (gold_ids or [])
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
        kept.append((example, len(teacher_ids), gold_ids))

    outputs = llm.generate(prompts, sampling_params)

    rows = {field: [] for field in args.student_fields}
    for (example, n_teacher, gold_ids), output in zip(kept, outputs):
        if gold_ids is not None:
            # prompt_logprobs[i] is the teacher's distribution over token i
            # given tokens < i -- the same meaning as gen.logprobs[i] for a
            # generated token -- so the gold positions slot straight in.
            gen_token_ids = gold_ids
            positions = output.prompt_logprobs[n_teacher : n_teacher + len(gold_ids)]
        else:
            gen = output.outputs[0]
            # A generation cut off by max_tokens has no <end_of_turn>, so the
            # length relationship the KD plugin relies on no longer holds.
            if gen.finish_reason != "stop" or not gen.logprobs:
                dropped["truncated"] += 1
                continue
            gen_token_ids = list(gen.token_ids)
            positions = gen.logprobs

        text = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
        teacher_logprobs, target_token_ids = topk_rows(positions, args.top_k)

        # The plugin derives one top-k width for the row from the *least
        # frequent* row length, so a single short row collapses the whole
        # sample to that width. Drop the sample instead.
        if any(len(row) != args.top_k for row in teacher_logprobs):
            dropped["short_topk"] += 1
            continue

        # A row is kept only if it validates for EVERY student variant, so the
        # two datasets stay row-for-row comparable.
        built = {
            field: build_student_row(
                tokenizer, example[field], gen_token_ids,
                teacher_logprobs, target_token_ids, text,
                args.top_k, args.max_seq_len,
            )
            for field in args.student_fields
        }
        if any(r is None for r in built.values()):
            dropped["retokenised"] += 1
            continue
        for field, row in built.items():
            rows[field].append(row)

    n_kept = len(next(iter(rows.values())))
    print(f"[{split_name}] kept {n_kept}/{len(split)} (dropped: {dropped})")
    return rows


def main():
    args = parse_args()
    if args.top_k < 2:
        raise ValueError(f"--top-k must be at least 2, got {args.top_k}")

    prompts_ds = load_from_disk(str(args.dataset_dir))
    # A plain Dataset is treated as a lone train split, so the output still has
    # the <name>/train layout the Axolotl configs point at.
    if isinstance(prompts_ds, Dataset):
        prompts_ds = DatasetDict({"train": prompts_ds})
    first_split = next(iter(prompts_ds.values()))
    needed = [args.teacher_field, *args.student_fields]
    if args.targets == "gold":
        needed.append(args.gold_field)
    missing = [f for f in needed if f not in first_split.column_names]
    if missing:
        raise ValueError(f"{args.dataset_dir} has no column(s) {missing}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_ids = stop_token_ids(args.model, tokenizer)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_seq_len,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.targets == "gold":
        # Score only: the one generated token is discarded.
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=1, prompt_logprobs=args.top_k,
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=args.top_k,
        )

    per_dataset = {field: {} for field in args.student_fields}
    for split in prompts_ds:
        rows = generate_split(
            llm, tokenizer, sampling_params, prompts_ds[split], split, args, stop_ids
        )
        for field, split_rows in rows.items():
            per_dataset[field][split] = Dataset.from_list(split_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for field, splits in per_dataset.items():
        out = args.output_dir / output_name(args.prefix, field)
        DatasetDict(splits).save_to_disk(str(out))
        source_record = provenance.read(args.dataset_dir)
        provenance.write(
            out, __file__, source_record["coverage"] if source_record else "unknown",
            DatasetDict(splits), args,
            teacher_model=args.model,
            targets=args.targets,
            student_field=field,
            source_dataset=os.path.abspath(args.dataset_dir),
            source_provenance=source_record,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
