# Runs the teacher model over the XSum prompts (from "prepare_data XSum.py")
# and records its generations plus per-token top-k logprobs, in the format
# expected by Axolotl's KD plugin (axolotl.integrations.kd.chat_template).
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEACHER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TOP_K = 20  # must stay fixed across the whole dataset
TEMPERATURE = 0.7
MAX_TOKENS = 128


def build_prompt_text(tokenizer, example):
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=True
    )


def generate_split(llm, tokenizer, sampling_params, split):
    prompt_texts = [build_prompt_text(tokenizer, example) for example in split]
    outputs = llm.generate(prompt_texts, sampling_params)

    rows = []
    for example, output in zip(split, outputs):
        gen = output.outputs[0]

        teacher_logprobs = []
        target_token_ids = []
        for position_logprobs in gen.logprobs:
            entries = list(position_logprobs.items())[:TOP_K]
            teacher_logprobs.append([logprob.logprob for _, logprob in entries])
            target_token_ids.append([token_id for token_id, _ in entries])

        rows.append({
            "messages_combined": example["messages"] + [
                {"role": "assistant", "content": gen.text}
            ],
            "teacher_logprobs": teacher_logprobs,
            "target_token_ids": target_token_ids,
            "temperature": TEMPERATURE,
        })
    return rows


def main():
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    llm = LLM(model=TEACHER_MODEL)

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        logprobs=TOP_K,
    )

    prompts_ds = load_from_disk("../datasets/xsum_prompts")

    kd_ds = DatasetDict({
        split: Dataset.from_list(
            generate_split(llm, tokenizer, sampling_params, prompts_ds[split])
        )
        for split in prompts_ds
    })

    kd_ds.save_to_disk("../datasets/xsum_kd_data")


if __name__ == "__main__":
    main()
