"""Evaluate a (optionally LoRA-adapted) causal LM on the XSum test set.

The summarisation counterpart to evaluate.py. Same interface - the same
--model / --adapter / azureml: reference handling, the same "one JSON per model
variant" output contract - but scored with ROUGE against XSum's reference
summaries instead of classification metrics.

Alongside ROUGE it records output-length statistics. Those are not decoration:
the failure mode seen on the crime arm was a merged model drifting back
towards base-model behaviour, and on a generation task that shows up first as
length and sentence count creeping back towards the base model's chattier
style, before it is fully visible in ROUGE.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# evaluate.py sits next to this file and owns the Azure ML model-registry
# resolution. Importing it keeps that logic in one place; because the script
# directory leads sys.path, this always resolves to the local evaluate.py and
# never to the (uninstalled) Hugging Face `evaluate` package.
try:
    from evaluate import resolve_model_ref
except ImportError as exc:  # pragma: no cover - configuration error, not a code path
    raise ImportError(
        "Could not import resolve_model_ref from evaluate.py - run this script "
        "from the evaluate/ directory, e.g. `uv run python evaluate_summarisation.py`."
    ) from exc


ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def load_model(model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Decoder-only batched generation requires left padding: with right
    # padding the pad tokens sit between the prompt and the first generated
    # token, and short prompts in a batch produce garbage.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_summaries(model, tokenizer, prompts, max_new_tokens, batch_size):
    summaries = []
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="generating",
        unit="batch",
    ):
        batch = prompts[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch
        ]
        # apply_chat_template already emits BOS, so don't let the tokenizer add
        # a second one.
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_length = inputs["input_ids"].shape[1]
        for row in output[:, input_length:]:
            summaries.append(
                tokenizer.decode(row, skip_special_tokens=True).strip()
            )
    return summaries


def count_sentences(text):
    """Rough sentence count - enough to detect one-sentence vs paragraph."""
    return sum(text.count(mark) for mark in ".!?") or (1 if text else 0)


def score_summaries(predictions, references):
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    per_example = []
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        per_example.append(
            {rouge_type: scores[rouge_type].fmeasure for rouge_type in ROUGE_TYPES}
        )

    metrics = {
        rouge_type: statistics.fmean(row[rouge_type] for row in per_example)
        for rouge_type in ROUGE_TYPES
    }
    prediction_words = [len(p.split()) for p in predictions]
    reference_words = [len(r.split()) for r in references]
    metrics.update(
        {
            "num_examples": len(predictions),
            "num_empty": sum(1 for p in predictions if not p),
            "mean_pred_words": statistics.fmean(prediction_words),
            "mean_ref_words": statistics.fmean(reference_words),
            "mean_pred_sentences": statistics.fmean(
                count_sentences(p) for p in predictions
            ),
        }
    )
    return metrics, per_example


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a (optionally LoRA-adapted) causal LM as a summariser on "
            "the XSum test set."
        )
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        help=(
            "Path, HF hub id, or 'azureml:<name>:<version>' reference to an "
            "Azure ML registered model, of the base/full model to load."
        ),
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Optional path or 'azureml:<name>:<version>' reference to a "
            "PEFT/LoRA adapter to apply on top of --model."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="../../datasets/xsum_dataset/test",
        help="Path to a dataset saved with `save_to_disk` (e.g. the test split).",
    )
    parser.add_argument(
        "--resource-group",
        default="tire-1",
        help="Azure resource group holding the Azure ML workspace (for azureml: refs).",
    )
    parser.add_argument(
        "--workspace-name",
        default="tire-2",
        help="Azure ML workspace name (for azureml: refs).",
    )
    parser.add_argument(
        "--subscription",
        default=None,
        help="Azure subscription id, if it isn't already az cli's active default.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=".azureml_models",
        help="Where azureml: model/adapter downloads are cached.",
    )
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help=(
            "Column holding the fully rendered user prompt. prepare_xsum_data.py "
            "writes this so training and evaluation cannot drift apart."
        ),
    )
    parser.add_argument("--reference-column", default="summary")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts per generate() call. Lower it if you hit CUDA OOM.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N examples (for quick runs).",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="Print this many prediction/reference pairs at the end (0 to disable).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If set, write the full results (metrics + generations) to this JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_from_disk(args.dataset)
    if args.limit is not None:
        dataset = dataset.select(range(args.limit))

    model_path = resolve_model_ref(
        args.model, args.model_cache_dir, args.resource_group,
        args.workspace_name, args.subscription,
    )
    adapter_path = resolve_model_ref(
        args.adapter, args.model_cache_dir, args.resource_group,
        args.workspace_name, args.subscription,
    )
    tokenizer, model = load_model(model_path, adapter_path)

    prompts = list(dataset[args.prompt_column])
    references = list(dataset[args.reference_column])
    predictions = generate_summaries(
        model, tokenizer, prompts, args.max_new_tokens, args.batch_size
    )
    metrics, per_example = score_summaries(predictions, references)

    print("Model      :", args.model)
    print("Adapter    :", args.adapter)
    print("Dataset    :", args.dataset)
    for rouge_type in ROUGE_TYPES:
        print(f"{rouge_type:11s}:", round(metrics[rouge_type], 4))
    print("Examples   :", metrics["num_examples"], f"({metrics['num_empty']} empty)")
    print(
        "Length     :",
        f"{metrics['mean_pred_words']:.1f} words / "
        f"{metrics['mean_pred_sentences']:.2f} sentences per prediction "
        f"(reference: {metrics['mean_ref_words']:.1f} words)",
    )

    for index in range(min(args.show, len(predictions))):
        print(f"\n--- example {index} ---")
        print("REFERENCE:", references[index])
        print("PREDICTED:", predictions[index])

    record = {
        "model": args.model,
        "model_path": model_path,
        "adapter": args.adapter,
        "adapter_path": adapter_path,
        "dataset": args.dataset,
        "metrics": metrics,
        # Kept per-example so two runs can be compared as a PAIRED sample -
        # see experiments/bootstrap_rouge.py. Comparing two mean ROUGE scores
        # as if they were independent throws away the pairing and badly
        # overstates the uncertainty.
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
