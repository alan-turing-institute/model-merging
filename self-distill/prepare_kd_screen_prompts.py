# Prompts for a self-distilled LoRA that matches the reference-trained LoRA in
# train/xsum_screen_gemma.yaml in everything except the training signal.
#
# Same rows as that run - the 4,000 training and 250 validation articles in
# datasets/xsum_screen/full, already cut to 512 words - with the three prompt
# columns prepare_data_XSum.py builds (teacher_messages, messages_long,
# messages_short), so generate_teacher_logprobs.py runs on them unchanged.
# The teacher therefore reads exactly the article text the student is trained
# and scored on.
#
# Usage, from self-distill/:
#   uv run python prepare_kd_screen_prompts.py ../datasets/xsum_screen/full ../datasets/xsum_kd_screen_prompts
import importlib.util
import sys
from pathlib import Path

from datasets import load_from_disk

spec = importlib.util.spec_from_file_location("prep", Path(__file__).with_name("prepare_data_XSum.py"))
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)

source, output = Path(sys.argv[1]), Path(sys.argv[2])
splits = load_from_disk(str(source)).remove_columns(["messages"])
prompts = splits.map(prep.convert_to_prompt)
prompts.save_to_disk(str(output))
print({name: len(split) for name, split in prompts.items()}, "->", output)
