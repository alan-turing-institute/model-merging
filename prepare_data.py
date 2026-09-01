# This file prepares the Reddit crime dataset for training.
# It loads the dataset, splits it into training, validation, and test sets,
# and converts the data into a format suitable for training a model.
from datasets import ClassLabel, DatasetDict, load_dataset

ds = load_dataset("Binaryy/crime_posts_reddit")

# Cast the label column to a ClassLabel type
train_dataset = ds["train"].cast_column(
    "label",
    ClassLabel(names=["non_crime", "crime"]),
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

# Convert the datasets to to a role/assistant format for training
def convert(example):
    label = "crime" if example["label"] == 1 else "not_crime"
    return {
        "messages": [
            {
                "role": "user",
                "content":
                    f"Is the following Reddit post about crime? Answer with "
                        f"exactly one word, either 'crime' or 'not_crime', and "
                        f"nothing else.\n\n" + f"{example['posts_name']}"
            },
            {
                "role": "assistant",
                "content": label
            }
        ]
    }

# Map the conversion function to the dataset and save it to disk
dataset = dataset.map(convert)
dataset.save_to_disk("../datasets/crime_dataset")

# Now split the training and validation datasets into two

# If we need to load the dataset from disk
# from datasets import load_from_disk
# dataset = load_from_disk("datasets/crime_dataset")

train = dataset["train"]
validation = dataset["validation"]

# Split the training set into two halves
train_split = train.train_test_split(
    test_size=0.5,
    seed=42,
    stratify_by_column="label"
)

train1 = train_split["train"]
train2 = train_split["test"]

# Split the validation sets into two halves
val_split = validation.train_test_split(
    test_size=0.5,
    seed=42,
    stratify_by_column="label"
)

validation1 = val_split["train"]
validation2 = val_split["test"]

dataset1 = DatasetDict({
    "train": train1,
    "validation": validation1
})

dataset2 = DatasetDict({
    "train": train2,
    "validation": validation2
})

# Map the conversion function to the dataset and save it to disk
# dataset = dataset.map(convert)

dataset1.save_to_disk("../datasets/crime_dataset1")
dataset2.save_to_disk("../datasets/crime_dataset2")