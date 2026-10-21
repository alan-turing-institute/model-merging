"""SummEval's 100 annotated CNN/DailyMail test articles, for comparing our models
with the systems the paper's experts rated (Fabbri et al., 2021, TACL).

    cd train && uv run python ../evaluate/prepare_summeval.py ../datasets/summeval

Joins the release's annotations (model_annotations.aligned.jsonl: per article
and system the decoded summary, 3 expert and 5 crowd ratings, 11 references) to
the article texts in mteb/summeval, by article id. Writes:

  <out>/pairs        one row per (article, system): the 16 rated systems plus
                     the original reference replayed as system "reference"
                     (unrated in the release); expert means per dimension
  <out>/cnndm_test   the 100 articles in our CNN/DailyMail test format, for
                     cnndm_task.py -T dataset_path=...

Documents are cut to the first 1,600 Gemma tokens exactly as
train/prepare_cnndm_data.py cuts ours, and every system - theirs and ours - is
judged against that same text. The reference is the original CNN/DailyMail
highlight (the first of the 11).
"""

import importlib.util
import json
import statistics
import sys
import urllib.request
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

ANNOTATIONS = "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl"
DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")
PREP = Path(__file__).resolve().parent.parent / "train" / "prepare_cnndm_data.py"


def main():
    out = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("prep", PREP)
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    tok = AutoTokenizer.from_pretrained(prep.TOKENIZER)

    with urllib.request.urlopen(ANNOTATIONS) as fh:
        records = [json.loads(line) for line in fh.read().decode().splitlines() if line.strip()]
    texts = {r["id"]: r["text"] for r in load_dataset("mteb/summeval", split="test")}
    missing = {r["id"] for r in records} - set(texts)
    if missing:
        raise SystemExit(f"{len(missing)} annotated articles have no text in mteb/summeval, e.g. {sorted(missing)[:3]}")

    articles = {}
    for r in records:
        if r["id"] not in articles:
            shaped = prep.shape({"id": r["id"], "article": texts[r["id"]], "highlights": r["references"][0]}, tok)
            articles[r["id"]] = shaped
    rows = []
    for r in records:
        experts = r["expert_annotations"]
        rows.append({"id": r["id"], "system": r["model_id"], "document": articles[r["id"]]["document"],
                     "summary": r["decoded"].strip(), "reference": articles[r["id"]]["summary"],
                     **{d: statistics.fmean(e[d] for e in experts) for d in DIMENSIONS},
                     "expert_annotations": json.dumps(experts), "references": json.dumps(r["references"])})
    for a in articles.values():
        rows.append({"id": a["id"], "system": "reference", "document": a["document"], "summary": a["summary"],
                     "reference": a["summary"], **{d: None for d in DIMENSIONS},
                     "expert_annotations": "[]", "references": "[]"})
    Dataset.from_list(rows).save_to_disk(str(out / "pairs"))
    Dataset.from_list(list(articles.values())).save_to_disk(str(out / "cnndm_test"))
    systems = sorted({r["system"] for r in rows})
    print(f"{len(articles)} articles, {len(rows)} (article, system) rows, systems: {' '.join(systems)}")


if __name__ == "__main__":
    main()
