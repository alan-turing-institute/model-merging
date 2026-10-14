"""Harder dialogue tasks for the summarisation arm, and their automatic checks.

SAMSum's plain summaries are learnt from ~100 examples, so experts trained on
halves of the data match the full-data model and a merge has nothing to
recover. These two tasks keep SAMSum's dialogues but ask for more:

  constrained   summarise in EXACTLY n words without using any form of k
                banned words (the summary's own topic words). Checked by
                program; the teacher's targets can be filtered to those that
                pass, so the ceiling is a verified teacher.
  constrained2  the same with the length relaxed to n +/- 2 words, stated
                as a range in the prompt; exact counts proved unlearnable from
                ~2,000 verified examples.
  xsumc         the constrained task on XSum articles with a relative
                tolerance: n is drawn around the article's own reference length
                (0.8-1.2x, 10-40 words) and the range is n +/- 10%, so short and
                long summaries are held to the same proportional standard.
  reorder       the turns are shuffled and labelled; reply with the original
                order as a JSON list of the labels. The gold order is known,
                so no teacher is needed.

Stdlib only: imported by the vLLM generator (dialogue_pilot.py) and run as a
scorer under evaluate's venv, which adds ROUGE and the semantic encoder:

    ../evaluate/.venv/bin/python dialogue_tasks.py score <pilot output dir>
"""

import json
import random
import re
import statistics
import sys
from pathlib import Path

MIN_TURNS, MAX_TURNS = 6, 14
N_BANNED = 3
STOPWORDS = set("""a about after again all also an and any are as at be because been before
being but by can could did do does doing down during each even for from get gets getting
going gonna got had has have having he her here him his how i if in into is it its just
let like make makes made me more most my need no not now of off on one only or other our
out over really said say says see she should so some still such than that the their them
then there these they this those through to today tomorrow too up very wants want was we
well were what when where which while who why will with would yesterday you your""".split())
WORD = re.compile(r"[a-z]+")


def turns(dialogue: str) -> list[str]:
    return [t.strip() for t in dialogue.replace("\r\n", "\n").split("\n") if t.strip()]


def speakers(dialogue: str) -> set[str]:
    return {t.split(":", 1)[0].strip().lower() for t in turns(dialogue) if ":" in t}


def word_count(text: str) -> int:
    """Words are whitespace-separated tokens holding a letter or digit."""
    return sum(bool(re.search(r"[A-Za-z0-9]", w)) for w in text.split())


def forms(word: str) -> set[str]:
    """Candidate base forms of a word under regular English inflection.

    Two words count as the same word when their candidate sets meet, so
    cookies/cookie/Cookie's, baked/baking/bake and party/parties all match.
    Candidates shorter than four letters are dropped, except the word itself,
    so bring/bred do not meet at "br". Irregular forms (bring/brought) are
    not caught.
    """
    x = re.sub(r"['\u2019]s$", "", word.lower())
    c = {x}
    if x.endswith("s") and not x.endswith("ss"):
        c.add(x[:-1])
    if x.endswith("es"):
        c.add(x[:-2])
    if x.endswith("ies"):
        c.add(x[:-3] + "y")
    if x.endswith("ed"):
        c |= {x[:-1], x[:-2]}
        if len(x) > 4 and x[-3] == x[-4]:
            c.add(x[:-3])
    if x.endswith("ing"):
        c |= {x[:-3], x[:-3] + "e"}
        if len(x) > 5 and x[-4] == x[-5]:
            c.add(x[:-4])
    return {f for f in c if len(f) >= 4} | {x}


def uses(text: str, word: str) -> bool:
    target = forms(word)
    return any(forms(token) & target for token in re.findall(r"[A-Za-z'\u2019]+", text))


def topic_words(summary: str, dialogue: str, k: int = N_BANNED) -> list[str]:
    """The summary's k most frequent content words that the dialogue also uses."""
    names, said = speakers(dialogue), set(WORD.findall(dialogue.lower()))
    counts: dict[str, int] = {}
    for w in WORD.findall(summary.lower()):
        if len(w) >= 4 and w not in STOPWORDS and w not in names and w in said:
            counts[w] = counts.get(w, 0) + 1
    return sorted(counts, key=lambda w: (-counts[w], summary.lower().index(w)))[:k]


def eligible(row) -> bool:
    return (MIN_TURNS <= len(turns(row["dialogue"])) <= MAX_TURNS
            and len(topic_words(row["summary"], row["dialogue"])) == N_BANNED)


def splits(samsum) -> dict[str, list]:
    """Eligible SAMSum rows: train shuffled once (arms are prefixes of it), 100
    validation rows for eval loss, and the other 518 held-out rows as test."""
    train = [r for r in samsum["train"] if eligible(r)]
    random.Random(0).shuffle(train)
    val = [r for r in samsum["validation"] if eligible(r)]
    return {"train": train, "validation": val[:100],
            "test": val[100:] + [r for r in samsum["test"] if eligible(r)]}


def splits_for(task: str, samsum_path: str) -> dict[str, list]:
    """Train/validation/test rows for a task. XSum tasks read a 12,000-article
    pool (prepare_xsum_screen_data.py --train-size 12000, beside samsum_raw);
    everything else is SAMSum dialogues."""
    from datasets import load_from_disk
    if not task.startswith("xsum"):
        return splits(load_from_disk(samsum_path))
    pool = Path(samsum_path).parent / "xsum_pool"
    full, test = load_from_disk(str(pool / "full")), load_from_disk(str(pool / "test"))
    ok = lambda r: len(topic_words(r["summary"], r["document"])) == N_BANNED
    train = [r for r in full["train"] if ok(r)]
    random.Random(0).shuffle(train)
    return {"train": train, "validation": [r for r in full["validation"] if ok(r)][:100],
            "test": [r for r in test if ok(r)]}


def xsum_constrained_example(row) -> dict:
    """XSum version of constrained2 with a relative tolerance (+/- 10% of n)."""
    rng = random.Random(f"xsumc-{row['id']}")
    ref = word_count(row["summary"])
    n = min(40, max(10, round(ref * rng.uniform(0.8, 1.2))))
    tol = max(1, round(0.1 * n))
    banned = topic_words(row["summary"], row["document"])
    prompt = (f"Summarise the article below in {n - tol} to {n + tol} words. Do not use any of these "
              f"words, in any form: {', '.join(banned)}. Words are counted as whitespace-separated "
              f"tokens. Reply with the summary only.\n\n{row['document'].strip()}")
    return {"id": row["id"], "task": "xsumc", "prompt": prompt, "n_words": n, "tolerance": tol,
            "banned": banned, "reference": row["summary"], "dialogue": row["document"]}


def unigram_f1(prediction: str, reference: str) -> float:
    p, r = WORD.findall(prediction.lower()), WORD.findall(reference.lower())
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    return 0.0 if not common else 2 * common / (len(p) + len(r))


def constrained_example(row, tolerance: int = 0) -> dict:
    rng = random.Random(f"constrained-{row['id']}")
    n, banned = rng.randint(12, 30), topic_words(row["summary"], row["dialogue"])
    length = f"exactly {n} words" if not tolerance else f"{n - tolerance} to {n + tolerance} words"
    prompt = (f"Summarise the dialogue below in {length}. Do not use any of these words, "
              f"in any form: {', '.join(banned)}. Words are counted as whitespace-separated tokens. "
              f"Reply with the summary only.\n\n{row['dialogue'].strip()}")
    return {"id": row["id"], "task": "constrained2" if tolerance else "constrained",
            "prompt": prompt, "n_words": n, "tolerance": tolerance,
            "banned": banned, "reference": row["summary"], "dialogue": row["dialogue"]}


def reorder_example(row) -> dict:
    rng = random.Random(f"reorder-{row['id']}")
    original = turns(row["dialogue"])
    shown = list(range(len(original)))
    while shown == sorted(shown):
        rng.shuffle(shown)                      # shown[j] = original position of label j
    gold = sorted(range(len(shown)), key=lambda j: shown[j])
    lines = "\n".join(f"[{j}] {original[shown[j]]}" for j in range(len(shown)))
    prompt = ("The turns of this dialogue have been shuffled and labelled. Reply with the labels in "
              "the order the turns were originally spoken, as a JSON list of integers, and nothing "
              f"else.\n\n{lines}")
    return {"id": row["id"], "task": "reorder", "prompt": prompt, "gold": gold}


WORKING = ("Show your working for the task below instead of answering it: list the banned words, "
           "write a draft, number the draft's words, give the count and check for banned words.\n\n")


def working_example(ex: dict, target: str) -> dict:
    """Distilling step-by-step (Hsieh et al., 2023): a second training task on
    the same dialogue, asking for the rationale behind a verified target. Built
    by program from the target, so it needs no teacher; never used at test time."""
    words = [w for w in target.split() if re.search(r"[A-Za-z0-9]", w)]
    lo, hi = ex["n_words"] - ex.get("tolerance", 0), ex["n_words"] + ex.get("tolerance", 0)
    rationale = (f"Banned words: {', '.join(ex['banned'])}.\n"
                 f"Draft: {target}\n"
                 f"Words: {' '.join(f'[{i}] {w}' for i, w in enumerate(words, 1))}\n"
                 f"Count: {len(words)}, within {lo} to {hi}.\n"
                 f"Banned words used: none.")
    return {"messages": [{"role": "user", "content": WORKING + ex["prompt"]},
                         {"role": "assistant", "content": rationale}]}


RATIONALE_ASK = (
    "Below are the shuffled, labelled turns of a dialogue, followed by their correct original "
    "order. For each step of that order, give in at most 12 words why the next turn follows the "
    "previous one (it answers a question, replies to a greeting, continues a plan, and so on). "
    "Write one line per step as '[a] -> [b]: reason', in the given order, and nothing else.")
EXPLAIN = ("Explain the original order of the dialogue turns below instead of answering: one line "
           "per step as '[a] -> [b]: reason'.\n\n")


def reorder_rationale_request(ex: dict) -> str:
    """Teacher prompt for a rationale of the GOLD order (rationalisation: the
    teacher is shown the answer and explains it, so every kept rationale agrees
    with the label)."""
    lines = ex["prompt"].split("\n\n", 1)[1]
    return f"{RATIONALE_ASK}\n\n{lines}\n\nCorrect order: {json.dumps(ex['gold'])}"


def rationale_matches(text: str, gold: list[int]) -> bool:
    """True when the rationale's steps are exactly the consecutive pairs of the gold order."""
    steps = [(int(a), int(b)) for a, b in re.findall(r"\[(\d+)\]\s*->\s*\[(\d+)\]", text)]
    return steps == list(zip(gold, gold[1:]))


def reorder_working_example(ex: dict, rationale: str) -> dict:
    """Distilling step-by-step's rationale task: same dialogue, asked for the
    explanation. Never used at test time, which keeps the plain prompt."""
    return {"messages": [{"role": "user", "content": EXPLAIN + ex["prompt"].split("\n\n", 1)[1]},
                         {"role": "assistant", "content": rationale}]}


def check_constrained(ex: dict, prediction: str) -> dict:
    count = word_count(prediction)
    used = [w for w in ex["banned"] if uses(prediction, w)]
    length_ok = abs(count - ex["n_words"]) <= ex.get("tolerance", 0)
    return {"words": count, "length_ok": length_ok, "word_error": abs(count - ex["n_words"]),
            "banned_used": used, "banned_ok": not used,
            "success": length_ok and not used,
            "f1": unigram_f1(prediction, ex["reference"])}


def kendall_tau(order: list[int], gold: list[int]) -> float:
    rank = {label: i for i, label in enumerate(order)}
    pairs = [(a, b) for i, a in enumerate(gold) for b in gold[i + 1:]]
    concordant = sum(rank[a] < rank[b] for a, b in pairs)
    return (2 * concordant - len(pairs)) / len(pairs)


def check_reorder(ex: dict, prediction: str) -> dict:
    match = re.search(r"\[[\d,\s]*\]", prediction)
    try:
        order = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        order = None
    valid = isinstance(order, list) and sorted(order) == sorted(ex["gold"])
    return {"valid": valid, "exact": valid and order == ex["gold"],
            "tau": kendall_tau(order, ex["gold"]) if valid else 0.0,
            "first_ok": valid and order[0] == ex["gold"][0]}


BUILDERS = {"constrained": constrained_example,
            "constrained2": lambda row: constrained_example(row, tolerance=2),
            "xsumc": xsum_constrained_example,
            "reorder": reorder_example}
CHECKS = {"constrained": check_constrained, "constrained2": check_constrained, "xsumc": check_constrained,
          "reorder": check_reorder}


def score(out_dir: Path) -> None:
    """Summarise every <model>.jsonl in out_dir; ROUGE and semantic for constrained."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluate"))
    from rouge_score import rouge_scorer
    import semantic_metrics
    rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    print(f"{'model':<28}{'task':<13}{'n':>5}  results")
    for path in sorted(out_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for task in ("constrained", "reorder"):
            got = [r for r in rows if r["task"] == task]
            if not got:
                continue
            mean = lambda key: statistics.fmean(float(r["check"][key]) for r in got)
            if task == "constrained":
                preds, refs = [r["prediction"] for r in got], [r["reference"] for r in got]
                sem, _ = semantic_metrics.score_all(preds, refs, [r["dialogue"] for r in got])
                r1 = statistics.fmean(rouge.score(a, b)["rouge1"].fmeasure for a, b in zip(refs, preds))
                line = (f"success {mean('success'):.3f}  exact length {mean('length_ok'):.3f}  "
                        f"no banned {mean('banned_ok'):.3f}  |words-n| {mean('word_error'):.1f}  "
                        f"rouge1 {r1:.3f}  semantic {sem.get('semantic', float('nan')):.3f}")
                best = [r for r in got if "best_of" in r]
                if best:
                    line += (f"  | best-of-{best[0]['best_of']} yield "
                             f"{statistics.fmean(r['best_ok'] for r in best):.3f}")
            else:
                line = (f"exact {mean('exact'):.3f}  tau {mean('tau'):.3f}  "
                        f"first turn {mean('first_ok'):.3f}  valid {mean('valid'):.3f}")
            print(f"{path.stem:<28}{task:<13}{len(got):>5}  {line}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        score(Path(sys.argv[2]))
