"""Prompts for gold-target KD on a full training arm (CLINC150 or EN->DE).

The assistant turn is the gold answer; the untrained 4B teacher only scores it
(generate_teacher_logprobs.py --targets gold), under a richer prompt than the
student's, as the BBC-style prompt did for XSum:

  clinc150   the teacher's prompt lists all 150 intent labels (untrained
             accuracy 0.704, against 0.051 with the student's short prompt)
  mt_domain  the student's prompt plus a system instruction to answer with the
             German translation only

    python prepare_gold_kd_prompts.py clinc150    # -> ../datasets/clinc150_kd_prompts
"""
import importlib.util
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

TASK = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent
MT_SYSTEM = ("You are a professional English-to-German translator. Reply with the German "
             "translation only, with no notes, labels or explanation.")


def clinc_labels_prompt():
    spec = importlib.util.spec_from_file_location("p", ROOT / "train" / "prepare_clinc150_data.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return lambda r: mod.convert_to_prompt(r)["messages_labels"]


teacher = clinc_labels_prompt() if TASK == "clinc150" else (
    lambda r: [{"role": "system", "content": MT_SYSTEM}, r["messages"][0]])
arm = load_from_disk(str(ROOT / "datasets" / TASK / "full"))
out = DatasetDict({
    split: Dataset.from_list([{"id": f"{split}-{i}", "summary": r["messages"][1]["content"],
                               "teacher_messages": teacher(r), "messages_short": [r["messages"][0]]}
                              for i, r in enumerate(arm[split])])
    for split in ("train", "validation")
})
dest = ROOT / "datasets" / f"{TASK}_kd_prompts"
out.save_to_disk(str(dest))
print(f"{dest}: " + ", ".join(f"{s} {len(d)}" for s, d in out.items()))
