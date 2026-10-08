# This file prepares the Reddit crime dataset for training.
# It loads the dataset, splits it into training, validation, and test sets,
# and converts the data into a format suitable for training a model.
from datasets import ClassLabel, DatasetDict, load_dataset

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
    ds = load_dataset("Binaryy/crime_posts_reddit")

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
    dataset.save_to_disk("../datasets/crime_dataset")


if __name__ == "__main__":
    main()
