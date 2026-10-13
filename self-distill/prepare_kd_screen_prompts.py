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
# --teacher-prompt bbc replaces the teacher's prompt with the BBC-style one that
# won the comparison in teacher_prompt_sweep.py (results/teacher-prompt-sweep.json:
# ROUGE-1 0.240 -> 0.269, entity support 0.843 -> 0.885 on validation). The
# student's prompts are unchanged either way.
#
# Usage, from self-distill/:
#   uv run python prepare_kd_screen_prompts.py ../datasets/xsum_screen/full ../datasets/xsum_kd_screen_prompts
#   uv run python prepare_kd_screen_prompts.py ../datasets/xsum_screen/full ../datasets/xsum_kd_screen_bbc_prompts --teacher-prompt bbc
import argparse
import importlib.util
from pathlib import Path

from datasets import load_from_disk

spec = importlib.util.spec_from_file_location("prep", Path(__file__).with_name("prepare_data_XSum.py"))
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)

# The one definition of the BBC-style teacher prompt; teacher_prompt_sweep.py
# imports it, so the prompt trained on is the prompt that was compared.
BBC_SYSTEM = (
    "You write the one-sentence summary that appears above a BBC News article. It states "
    "the story's main news in about 20 words, the way BBC summaries do: usually in the "
    "past or present perfect tense (for example 'has died', 'were jailed', 'has been "
    "approved'), naming the key people, places, organisations and numbers. Reply with "
    "the sentence only, in plain text, with no preamble, label or quotation marks."
)


def bbc_teacher_messages(document):
    return [{"role": "system", "content": BBC_SYSTEM},
            {"role": "user", "content": f"Article:\n\n{document}\n\nSummarise the article above in one sentence of about 20 words."}]


def main():
    parser = argparse.ArgumentParser(description="Prompts for the self-distilled XSum LoRA.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--teacher-prompt", choices=["current", "bbc"], default="current")
    args = parser.parse_args()
    splits = load_from_disk(str(args.source)).remove_columns(["messages"])
    prompts = splits.map(prep.convert_to_prompt)
    if args.teacher_prompt == "bbc":
        prompts = prompts.map(lambda row: {"teacher_messages": bbc_teacher_messages(row["document"])})
    prompts.save_to_disk(str(args.output))
    print({name: len(split) for name, split in prompts.items()}, f"teacher prompt: {args.teacher_prompt}", "->", args.output)


if __name__ == "__main__":
    main()
