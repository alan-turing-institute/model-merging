"""Run the Inspect AI XSum task and write this repo's results JSON.

Inspect owns the evaluation; this wrapper owns the two things Inspect does not
know about:

1. `azureml:<name>:<version>` model references, resolved the same way as in
   evaluate.py, so the pipeline can name registry artifacts.
2. The results JSON contract the analysis tools already read -
   experiments/bootstrap_rouge.py (per-example paired bootstrap) and
   experiments/crosstask_table.py (joint retention table). Switching to Inspect
   should not cost the analysis that is already built on top of those files.

Both outputs are produced: the native `.eval` log (browse it with
`inspect view --log-dir <dir>`) and the JSON. The JSON records the log path, so
a number in a table can always be traced back to its samples.

    uv run python run_inspect_xsum.py \
      --model google/gemma-3-4b-it \
      --adapter azureml:gemma3-xsum-full-lora:3 \
      --output results/gemma3-xsum-full-lora.json
"""

import argparse
import json
import os
import statistics
from pathlib import Path

# MUST precede the inspect_ai eval import path being exercised: Inspect resolves
# the --model string before it imports the task file, so a provider registered
# only inside xsum_task.py is registered too late and `hf-peft/...` comes back
# as "Model API hf-peft not recognized".
import hf_peft_provider  # noqa: F401
from inspect_ai import eval as inspect_eval

from evaluate import resolve_model_ref

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", default="../../datasets/xsum_dataset/test")
    parser.add_argument("--resource-group", default="tire-1")
    parser.add_argument("--workspace-name", default="tire-2")
    parser.add_argument("--subscription", default=None)
    parser.add_argument("--model-cache-dir", default=".azureml_models")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda:0, cpu, ... (auto by default)")
    parser.add_argument("--log-dir", default="logs", help="Where Inspect writes .eval logs.")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", type=int, default=3)
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

    # Greedy decoding, matching evaluate_summarisation.py. Inspect's HF provider
    # defaults do_sample to TRUE, so leaving it unset would quietly make every
    # comparison in this repo noisier than the one it is being compared against.
    model_args = {"batch_size": args.batch_size, "do_sample": False}
    if args.device:
        model_args["device"] = args.device
    if adapter_path:
        model_args["adapter_path"] = adapter_path

    # A local directory is loaded via model_path; model_name is then only a label.
    if os.path.isdir(model_path):
        model_args["model_path"] = model_path
        model_name = Path(model_path).name
    else:
        model_name = model_path

    provider = "hf-peft" if adapter_path else "hf"

    task_args = {"dataset_path": args.dataset}
    if args.limit is not None:
        task_args["limit"] = args.limit

    log = inspect_eval(
        # A bare filename, not a Path and not an absolute path: Inspect's task
        # loader iterates the value as a list of specs (a Path raises TypeError)
        # and globs it relative to a root dir (an absolute path raises
        # NotImplementedError). Run this from evaluate/, as with the other
        # scripts here - they already depend on cwd for .azureml_models and for
        # importing evaluate.py.
        "xsum_task.py",
        model=f"{provider}/{model_name}",
        model_args=model_args,
        task_args=task_args,
        max_tokens=args.max_new_tokens,
        log_dir=args.log_dir,
    )[0]

    if log.status != "success":
        raise SystemExit(f"Inspect eval failed: {log.error}")

    # Sample order is not guaranteed to match dataset order - samples can
    # complete out of order. bootstrap_rouge.py pairs runs by index, so sort by
    # sample id to make the ordering identical across every run of this dataset.
    samples = sorted(log.samples, key=lambda s: str(s.id))

    predictions, references, per_example = [], [], []
    for sample in samples:
        score = sample.scores["rouge"].value
        predictions.append(sample.output.completion.strip())
        references.append(sample.target if isinstance(sample.target, str) else sample.target[0])
        per_example.append({r: float(score[r]) for r in ROUGE_TYPES})

    metrics = {r: statistics.fmean(row[r] for row in per_example) for r in ROUGE_TYPES}
    metrics.update(
        {
            "num_examples": len(predictions),
            "num_empty": sum(1 for p in predictions if not p),
            "mean_pred_words": statistics.fmean(
                float(s.scores["rouge"].value["pred_words"]) for s in samples
            ),
            "mean_ref_words": statistics.fmean(len(r.split()) for r in references),
            "mean_pred_sentences": statistics.fmean(
                float(s.scores["rouge"].value["pred_sentences"]) for s in samples
            ),
            # Inspect computes this and the hand-rolled script never did: the
            # standard error on mean ROUGE-1, i.e. how much of a gap between two
            # models is worth taking seriously at all.
            "rouge1_stderr": next(
                (
                    s.metrics["stderr"].value
                    for s in log.results.scores
                    if s.name == "rouge1" and "stderr" in s.metrics
                ),
                None,
            ),
        }
    )

    print("Model      :", args.model)
    print("Adapter    :", args.adapter)
    print("Dataset    :", args.dataset)
    for rouge_type in ROUGE_TYPES:
        print(f"{rouge_type:11s}:", round(metrics[rouge_type], 4))
    if metrics["rouge1_stderr"] is not None:
        print("R-1 stderr :", round(metrics["rouge1_stderr"], 4))
    print("Examples   :", metrics["num_examples"], f"({metrics['num_empty']} empty)")
    print(
        "Length     :",
        f"{metrics['mean_pred_words']:.1f} words / "
        f"{metrics['mean_pred_sentences']:.2f} sentences per prediction "
        f"(reference: {metrics['mean_ref_words']:.1f} words)",
    )
    print("Inspect log:", log.location)

    for index in range(min(args.show, len(predictions))):
        print(f"\n--- example {index} ---")
        print("REFERENCE:", references[index])
        print("PREDICTED:", predictions[index])

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
        "references": references,
        "per_example_rouge": per_example,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2))
        print("\nSaved results to", output_path)

    return record


if __name__ == "__main__":
    main()
