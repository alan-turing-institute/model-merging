# This file prepares the XSum dataset for self-distillation training.
# It loads the dataset, samples down to a manageable size, and converts
# the articles into user-turn prompts ready for teacher generation.
from datasets import DatasetDict, load_dataset

ds = load_dataset("EdinburghNLP/xsum")

# The dataset already has a split into train, validation, and test sets, so we can just use those directly.

# This has 200k rows, so we will sample 10k rows for training, 1k for validation, and 1k for testing.
sample_sizes = {"train": 10_000, "validation": 1_000, "test": 1_000}

small_ds = DatasetDict({
    split: ds[split].shuffle(seed=42).select(range(n))
    for split, n in sample_sizes.items()
})

# Convert the datasets to a role format for self-distillation training of the teacher.
def convert_to_prompt(example):
    return {
        "messages": [
            {
                "role": "user",
                "content":
                    f"Please summarise the following in one sentence:"
                        f"\n\n" + f"{example['document']}"
            }
        ]
    }

# Map the conversion function to the dataset and save the prompts to disk.
# generate_teacher_logprobs XSum.py loads this to run teacher generation.
prompts_ds = small_ds.map(convert_to_prompt)
prompts_ds.save_to_disk("../datasets/xsum_prompts")
