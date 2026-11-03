# Script to convert a LoRA adapter into a full model by merging it with the base model.
# The output gets a provenance.json (see provenance.py): the adapter's record,
# still kind "trained", with a "materialised" entry naming the adapter, the base
# and the dtype, and health checked on the merged weights. The script exits
# non-zero if those hold NaN/inf.
import argparse
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import provenance


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

    dtype = str(merged_model.dtype).removeprefix("torch.")
    record = provenance.write_materialised(output_path, __file__, lora_path, base_model_name, dtype)
    print(f"wrote {output_path}/{provenance.FILENAME} ({dtype})")
    if not record["health"]["ok"]:
        print(f"MODEL IS BROKEN: {record['health']['nonfinite_tensors']}/{record['health']['tensors']} "
              f"weight tensors hold NaN/inf (e.g. {record['health']['nonfinite_examples']})",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a base causal language model."
    )
    parser.add_argument("base_model", help="Base model name or local path")
    parser.add_argument("lora_path", help="LoRA adapter name or local path")
    parser.add_argument("output_path", help="Directory for the merged model")
    args = parser.parse_args()
    sys.exit(main(args.base_model, args.lora_path, args.output_path))
