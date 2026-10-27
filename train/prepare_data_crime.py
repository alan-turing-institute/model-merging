# This file prepares the Reddit crime dataset for training.
# It loads the dataset, splits it into training, validation, and test sets,
# and converts the data into a format suitable for training a model.
# A provenance.json (see ../provenance.py) records it as the full dataset.
import sys
from pathlib import Path

from datasets import ClassLabel, DatasetDict, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import provenance  # noqa: E402  (repo root, shared with split_dataset.py etc.)

SOURCE = "Binaryy/crime_posts_reddit"
OUTPUT_DIR = "../datasets/crime/crime_dataset"

PROMPT = (
    "Is the following Reddit post about crime? Answer with exactly one word, "
    "either 'crime' or 'not_crime', and nothing else."
)


# Convert the datasets to to a role/assistant format for training
def convert(example):
    label = "crime" if example["label"] == 1 else "not_crime"
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{PROMPT}\n\n{example['posts_name']}"
            },
            {
                "role": "assistant",
                "content": label
            }
        ]
    }


def main():
    ds = load_dataset(SOURCE)

    # Cast the label column to a ClassLabel type
    train_dataset = ds["train"].cast_column(
        "label",
        ClassLabel(names=["non_crime", "crime"]), # change the labels 0,1 to non_crime, crime
    )

    # Split off training first
    train_val_test = train_dataset.train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label"
    )

    train = train_val_test["train"]
    val_test = train_val_test["test"]

    # Split the validation and test sets
    val_test = val_test.train_test_split(
        test_size=0.5,
        seed=42,
        stratify_by_column="label"
    )

    validation = val_test["train"]
    test = val_test["test"]

    dataset = DatasetDict({
        "train": train,
        "validation": validation,
        "test": test
    })

    # Map the conversion function to the dataset and save it to disk
    dataset = dataset.map(convert)
    dataset.save_to_disk(OUTPUT_DIR)
    provenance.write(
        OUTPUT_DIR, __file__, "full", dataset,
        source=SOURCE,
        sampling="all rows; 80/10/10 train/validation/test, stratified by label, seed 42",
    )


if __name__ == "__main__":
    main()
