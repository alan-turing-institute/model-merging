"""Run the Inspect AI crime task and write this repo's results JSON.

The classification counterpart to run_inspect_xsum.py. Inspect owns the
evaluation; this wrapper owns azureml: reference resolution and the results
JSON contract that experiments/mcnemar.py (paired significance testing) and
experiments/crosstask_table.py (joint retention table) already read.

Metric definitions are reconstructed here exactly as evaluate.py computes them,
from the same sklearn functions and with the same rule that an unparsed
generation is scored wrong rather than dropped. A new evaluator that quietly
redefined precision would make every comparison with the pilot meaningless.

    uv run python run_inspect_crime.py \
      --model google/gemma-3-4b-it \
      --adapter azureml:gemma3-crime-1-of-2-lora:3 \
      --output results/half1.json
"""

import argparse
import json
import os
from pathlib import Path

# Must precede eval: Inspect resolves --model before importing the task file,
# so a provider registered only there is registered too late for hf-peft/...
import hf_peft_provider  # noqa: F401
from inspect_ai import eval as inspect_eval
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from evaluate import resolve_model_ref


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", default="../../datasets/crime_dataset/test")
    parser.add_argument("--resource-group", default=os.environ.get("AZUREML_RG", "tire-1"))
    parser.add_argument("--workspace-name", default=os.environ.get("AZUREML_WS", "tire-2"))
    parser.add_argument("--subscription", default=None)
    parser.add_argument("--model-cache-dir", default=".azureml_models")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help=(
            "Torch dtype for the weights. evaluate.py and "
            "evaluate_summarisation.py both load bfloat16; Inspect's HF provider "
            "pins nothing, so without this the two evaluators could differ in "
            "numerics for the same model."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_model_ref(
        args.model, args.model_cache_dir, args.resource_group,
        args.workspace_name, args.subscription,
    )
    adapter_path = resolve_model_ref(
        args.adapter, args.model_cache_dir, args.resource_group,
        args.workspace_name, args.subscription,
    )

    # do_sample=False to match evaluate.py's greedy decoding. Inspect's HF
    # provider defaults it to true.
    model_args = {"batch_size": args.batch_size, "do_sample": False}
    if args.dtype and args.dtype != "auto":
        model_args["dtype"] = args.dtype
    if args.device:
        model_args["device"] = args.device
    if adapter_path:
        model_args["adapter_path"] = adapter_path
    if os.path.isdir(model_path):
        model_args["model_path"] = model_path
        model_name = Path(model_path).name
    else:
        model_name = model_path

    task_args = {"dataset_path": args.dataset}
    if args.limit is not None:
        task_args["limit"] = args.limit

    log = inspect_eval(
        "crime_task.py",
        model=f"{'hf-peft' if adapter_path else 'hf'}/{model_name}",
        model_args=model_args,
        task_args=task_args,
        max_tokens=args.max_new_tokens,
        log_dir=args.log_dir,
    )[0]

    if log.status != "success":
        raise SystemExit(f"Inspect eval failed: {log.error}")

    # Sort by id: samples can complete out of order, and mcnemar.py pairs runs
    # by index. Without this the pairing would be silently wrong.
    samples = sorted(log.samples, key=lambda s: int(s.id))

    predictions, actuals = [], []
    for sample in samples:
        meta = sample.scores["crime_label"].metadata
        predictions.append(meta["predicted"])
        actuals.append(meta["actual"])

    # evaluate.py's rule, reproduced exactly: an unparsed generation (None) is
    # scored as the wrong label, so sklearn never sees a None and the
    # denominator stays the full test set.
    scored = [
        prediction if prediction is not None else 1 - actual
        for prediction, actual in zip(predictions, actuals)
    ]

    metrics = {
        "accuracy": accuracy_score(actuals, scored),
        "precision": precision_score(actuals, scored, zero_division=0),
        "recall": recall_score(actuals, scored, zero_division=0),
        "f1": f1_score(actuals, scored, zero_division=0),
        "confusion_matrix": confusion_matrix(actuals, scored).tolist(),
        "num_examples": len(actuals),
        "num_unparsed": sum(p is None for p in predictions),
    }

    print("Model    :", args.model)
    print("Adapter  :", args.adapter)
    print("Accuracy :", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall   :", round(metrics["recall"], 4))
    print("F1       :", round(metrics["f1"], 4))
    print("Unparsed :", metrics["num_unparsed"], "/", metrics["num_examples"])
    print("Confusion matrix:", metrics["confusion_matrix"])
    print("Inspect log:", log.location)

    record = {
        "model": args.model,
        "provenance": args.provenance,
        "model_path": model_path,
        "adapter": args.adapter,
        "adapter_path": adapter_path,
        "dataset": args.dataset,
        "eval_log": log.location,
        "metrics": metrics,
        "predictions": predictions,
        "actuals": actuals,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2))
        print("Saved results to", output_path)

    return record


if __name__ == "__main__":
    main()
