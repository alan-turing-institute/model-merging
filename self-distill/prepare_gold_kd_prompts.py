"""Prompts for gold-target KD on a training arm.

The assistant turn is the gold answer; the untrained 4B teacher only scores it
(generate_teacher_logprobs.py --targets gold), under a richer prompt than the
student's, as the BBC-style prompt did for XSum:

  clinc150       the teacher's prompt lists all 150 intent labels (untrained
                 accuracy 0.704, against 0.051 with the student's short prompt)
  mt_domain      the student's prompt plus a system instruction to answer with
                 the German translation only
  billsum_split  the student's prompt plus a house-style system instruction,
  cnndm          written the way BBC_SYSTEM is for XSum: what the dataset's
  samsum         references look like (CRS bill summaries, CNN/Daily Mail
                 highlights, SAMSum third-person summaries). Unlike the BBC
                 prompt, these were not compared against the plain prompt first.

    python prepare_gold_kd_prompts.py clinc150    # -> ../datasets/clinc150_kd_prompts

The summarisation arms are the ones with a measured half-to-full gap under SFT
where there is one: billsum_split/full (1,000; 500 -> 1,000 is +0.025 rougeLsum),
cnndm_curve/n500 (250 -> 500 is +0.020) and samsum/n800 (SAMSum is flat from
100 rows, so there it measures only whether merging costs anything).
"""
import importlib.util
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

TASK = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent
MT_SYSTEM = ("You are a professional English-to-German translator. Reply with the German "
             "translation only, with no notes, labels or explanation.")
STYLE = {
    "billsum_split": (
        "You write the summary of a US Congressional bill in the style of the Congressional "
        "Research Service. It opens with the bill's short title followed by ' - ' (for example "
        "'Fair Telephone Billing Act of 2012 - Amends the Communications Act of 1934 to ...'), "
        "then describes what each main provision does, in order, in sentences that begin with a "
        "present-tense verb and no subject ('Amends ...', 'Requires the Secretary of ...', "
        "'Directs ...', 'Prohibits ...'), naming the agencies, Acts and amounts involved. It is "
        "usually 100 to 250 words. Reply with the summary only, in plain text, with no preamble, "
        "headings or bullet points."),
    "cnndm": (
        "You write the story highlights that appear at the top of a CNN or Daily Mail online "
        "article: three or four short sentences, each on its own line, each stating one key fact "
        "from the story in about 10 to 20 words, reusing the article's own names, numbers, places "
        "and phrasing. Reply with the highlights only, in plain text, with no bullets, numbering, "
        "heading or preamble."),
    "samsum": (
        "You write the summary of a messenger conversation in the style of the SAMSum dataset: "
        "one or two short sentences in the third person, naming the participants and saying what "
        "they did, decided or will do (for example 'Amanda baked cookies and will bring Jerry "
        "some tomorrow.'). It is usually about 20 words and does not quote the chat. Reply with "
        "the summary only, in plain text, with no preamble or label."),
}
ARMS = {"clinc150": "clinc150/full", "mt_domain": "mt_domain/full", "billsum_split": "billsum_split/full",
        "cnndm": "cnndm_curve/n500", "samsum": "samsum/n800"}


def clinc_labels_prompt():
    spec = importlib.util.spec_from_file_location("p", ROOT / "train" / "prepare_clinc150_data.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return lambda r: mod.convert_to_prompt(r)["messages_labels"]


if TASK == "clinc150":
    teacher = clinc_labels_prompt()
else:
    system = MT_SYSTEM if TASK == "mt_domain" else STYLE[TASK]
    teacher = lambda r: [{"role": "system", "content": system}, r["messages"][0]]
arm = load_from_disk(str(ROOT / "datasets" / ARMS[TASK]))
out = DatasetDict({
    split: Dataset.from_list([{"id": f"{split}-{i}", "summary": r["messages"][1]["content"],
                               "teacher_messages": teacher(r), "messages_short": [r["messages"][0]]}
                              for i, r in enumerate(arm[split])])
    for split in ("train", "validation")
})
dest = ROOT / "datasets" / f"{TASK}_kd_prompts"
out.save_to_disk(str(dest))
print(f"{ARMS[TASK]} -> {dest}: " + ", ".join(f"{s} {len(d)}" for s, d in out.items()))
