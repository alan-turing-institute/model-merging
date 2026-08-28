# Script to convert a LoRA adapter into a full model by merging it with the base model.
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main(base_model_name, lora_path, output_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto"
    )
    model = PeftModel.from_pretrained(
        base_model,
        lora_path
    )
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a base causal language model."
    )
    parser.add_argument("base_model", help="Base model name or local path")
    parser.add_argument("lora_path", help="LoRA adapter name or local path")
    parser.add_argument("output_path", help="Directory for the merged model")
    args = parser.parse_args()
    main(args.base_model, args.lora_path, args.output_path)