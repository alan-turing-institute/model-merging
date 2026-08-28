import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_to_label(text):
    text = text.lower()
    if "not_crime" in text:
        return 0
    if text.strip().startswith("not"):
        return 0
    if "crime" in text:
        return 1
    return None


def load_model(model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model


def evaluate_model(
    model,
    tokenizer,
    dataset,
    text_column="posts_name",
    label_column="label",
    max_new_tokens=5,
):
    predictions = []
    actuals = []
    for row in dataset:
        messages = [
            {
                "role": "user",
                "content": (
                    "Is the following Reddit post about crime?\n\n"
                    + row[text_column]
                ),
            }
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_length:]
        response = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        predictions.append(convert_to_label(response))
        actuals.append(row[label_column])

    # An unparsed generation is scored as wrong (rather than dropped) so every
    # example counts towards the metrics below and sklearn never sees a None.
    scored_predictions = [
        pred if pred is not None else 1 - actual
        for pred, actual in zip(predictions, actuals)
    ]

    metrics = {
        "accuracy": accuracy_score(actuals, scored_predictions),
        "precision": precision_score(actuals, scored_predictions),
        "recall": recall_score(actuals, scored_predictions),
        "f1": f1_score(actuals, scored_predictions),
        "confusion_matrix": confusion_matrix(actuals, scored_predictions).tolist(),
        "num_examples": len(actuals),
        "num_unparsed": sum(pred is None for pred in predictions),
    }
    return {
        "metrics": metrics,
        "predictions": predictions,
        "actuals": actuals,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a (optionally LoRA-adapted) causal LM on the crime "
            "classification test set."
        )
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        help="Path or HF hub id of the base/full model to load.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional path to a PEFT/LoRA adapter to apply on top of --model.",
    )
    parser.add_argument(
        "--dataset",
        default="../../datasets/crime_dataset/test",
        help="Path to a dataset saved with `save_to_disk` (e.g. the test split).",
    )
    parser.add_argument("--text-column", default="posts_name")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N examples (for quick runs).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If set, write the full results (metrics + predictions) to this JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_from_disk(args.dataset)
    if args.limit is not None:
        dataset = dataset.select(range(args.limit))

    tokenizer, model = load_model(args.model, args.adapter)
    results = evaluate_model(
        model,
        tokenizer,
        dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        max_new_tokens=args.max_new_tokens,
    )

    metrics = results["metrics"]
    print("Model    :", args.model)
    print("Adapter  :", args.adapter)
    print("Dataset  :", args.dataset)
    print("Accuracy :", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall   :", metrics["recall"])
    print("F1       :", metrics["f1"])
    print("Unparsed :", metrics["num_unparsed"], "/", metrics["num_examples"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    record = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        **results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2))
        print("Saved results to", output_path)

    return record


if __name__ == "__main__":
    main()
