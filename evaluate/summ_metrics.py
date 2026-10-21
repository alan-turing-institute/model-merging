# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "inspect-ai>=0.3.260",
#     "datasets",
#     "torch==2.13.0",  # the build evaluate/ already runs on the GH200 nodes
#     "transformers",
#     "bert-score>=0.3.13",
#     "deepeval==4.3.1",
#     "openai",
#     "rouge-score",
#     "sacrebleu",
#     "nltk",
#     "pycocoevalcap",
# ]
# ///
"""BERTScore, DeepEval's SummarizationMetric and an Inspect LLM judge on finished summarisation evals.

    uv run --script summ_metrics.py logs --task xsum --models 'kdgold75|^base' \
        --metrics bertscore deepeval --judge-url http://127.0.0.1:8000/v1 --judge-model judge

summ_metrics.sbatch runs this with a vLLM server for the judge on the same GPU:
Qwen3-32B by default, outside the Gemma family it judges, so it has no
same-family preference for the students' style.
Its environment is its own (the inline metadata above, locked in
summ_metrics.py.lock): deepeval pins click, rich, tabulate and, through typer,
huggingface-hub below what the evaluate project uses, so it stays out of that lock.

Scores the generations already in the Inspect logs - nothing is regenerated -
for the latest successful log of each model on the task (labelled as in
merge_report.py: the adapter's or merged model's directory name, or `base`).
Per-sample results go to <out>/<task>/<label>.<metric>.jsonl and are reused on
a re-run, so an interrupted job resumes; <out>/<task>/summary.json holds the means.

BERTSCORE (Zhang et al., 2020): token-level cosine matching of contextual
embeddings between summary and reference, as precision, recall and F1. Default
roberta-large with baseline rescaling, the package's choice for English, so the
numbers are on the scale papers report. The encoder reads 510 tokens per side:
a long BillSum generation is scored on its first 510.

DEEPEVAL SummarizationMetric (deepeval.com/docs/metrics-summarization): an LLM
judge, reference-free. alignment = share of the summary's claims the judge finds
supported by facts extracted from the source (penalises hallucination and
contradiction); coverage = of n yes/no questions generated from the source that
the source answers "yes", the share the summary also answers "yes". Score =
min(alignment, coverage). The judge sees the document the model saw (the task's
`document`, already truncated as in training).

INSPECT LLM JUDGE (summ_judge.py): SummEval's four dimensions rated 1-5 by the
same judge, through Inspect's own scoring (inspect_ai.score_async on the log, the
judge as the `grader` model role). The judged logs are kept in <out>/<task>/judged/
for `inspect view`; relevance is rated against each dataset's expected length.

PAIRED ACROSS MODELS. Left alone, the metric re-derives the source's facts,
questions and answers for every summary, so two models' coverage would be
measured against different questions. Here those three document-side outputs are
made once per document and cached on disk (keyed by judge, n and the truths
limit), and every model's summary of that document is judged against the same
ones - the comparison is paired, and the judge's cost per extra model drops to
the three summary-side calls (claims, alignment verdicts, summary answers). The
reason text is skipped. This overrides private methods of deepeval 4.3.1, which
the script's metadata pins; check them before moving the pin.
"""

import argparse
import asyncio
import collections
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

from datasets import load_from_disk
from inspect_ai.log import list_eval_logs, read_eval_log

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

TASK_DATASETS = {"cnndm": Path(__file__).resolve().parent.parent / "datasets" / "cnndm" / "test"}


def label(log):
    system = (log.eval.task_args or {}).get("system")
    if system:  # summeval_task.py replays one SummEval system's summaries
        return f"summeval-{system}"
    args = log.eval.model_args or {}
    for key in ("adapter_path", "model_path"):
        if args.get(key):
            return Path(str(args[key]).rstrip("/")).name
    return "base-" + str(log.eval.model).split("/")[-1]


def latest_logs(log_dir, task, pattern, size, variant, dataset):
    logs = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info, header_only=True)
        if log.status != "success" or log.eval.task.split("/")[-1] != task:
            continue
        targs = log.eval.task_args or {}
        if variant and targs.get("prompt_variant", variant) != variant:
            continue
        if dataset and dataset not in str(targs.get("dataset_path", "")):
            continue
        replayed = bool(targs.get("system"))  # summeval_task.py: no model of ours
        if not replayed and f"gemma-3-{size}-it" not in str(log.eval.model) \
                and not (log.eval.model_args or {}).get("model_path"):
            continue
        name = label(log)
        if pattern and not re.search(pattern, name):
            continue
        if name not in logs or log.eval.created > logs[name][0].eval.created:
            logs[name] = (log, info)
    return {name: read_eval_log(info) for name, (_, info) in sorted(logs.items())}


def rows_of(log, task, limit):
    """(id, document, reference, prediction) for each sample."""
    docs = None
    out = []
    for s in log.samples[:limit] if limit else log.samples:
        doc = (s.metadata or {}).get("document")
        if doc is None:  # cnndm keeps only the row index; read the document back
            if docs is None:
                path = (log.eval.task_args or {}).get("dataset_path") or TASK_DATASETS[task]
                docs = load_from_disk(str(path))["document"]
            doc = docs[int(s.id)]
        target = s.target if isinstance(s.target, str) else "\n".join(s.target)
        out.append((str(s.id), doc, target, (s.output.completion or "").strip()))
    return out


def load_done(path):
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("error") is None}


def run_bertscore(rows, path, model):
    from bert_score import BERTScorer

    done = load_done(path)
    todo = [r for r in rows if r[0] not in done]
    if todo:
        scorer = run_bertscore.scorer = getattr(run_bertscore, "scorer", None) or BERTScorer(
            model_type=model, lang="en", rescale_with_baseline=True)
        p, r, f = scorer.score([t[3] for t in todo], [t[2] for t in todo], batch_size=32)
        with path.open("a") as fh:
            for t, pi, ri, fi in zip(todo, p.tolist(), r.tolist(), f.tolist()):
                rec = {"id": t[0], "bertscore_p": pi, "bertscore_r": ri, "bertscore_f1": fi}
                fh.write(json.dumps(rec) + "\n")
                done[t[0]] = rec
    return [done[t[0]] for t in rows]


def judge_class():
    from deepeval.models import DeepEvalBaseLLM
    from openai import AsyncOpenAI, OpenAI

    class OpenAICompatibleJudge(DeepEvalBaseLLM):
        """A judge behind an OpenAI-compatible endpoint (vLLM here), with JSON
        constrained to the metric's own pydantic schema - DeepEval's structured
        metrics fail on free-form output from open models."""

        def __init__(self, base_url, model, max_tokens):
            self.base_url, self.max_tokens = base_url, max_tokens
            super().__init__(model)

        def load_model(self):
            self.client = OpenAI(base_url=self.base_url, api_key="none")
            self.aclient = AsyncOpenAI(base_url=self.base_url, api_key="none", timeout=600)
            return self.client

        def get_model_name(self):
            return self.name

        def _request(self, prompt, schema):
            kw = {"model": self.name, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0, "max_tokens": self.max_tokens,
                  # Qwen3 opens every answer with a <think> block unless told not
                  # to, which a JSON-constrained reply cannot hold. Templates
                  # without the switch ignore it.
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            if schema is not None:
                kw["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "schema": schema.model_json_schema()}}
            return kw

        def generate(self, prompt, schema=None):
            text = self.client.chat.completions.create(**self._request(prompt, schema)).choices[0].message.content
            return schema.model_validate_json(text) if schema else text

        async def a_generate(self, prompt, schema=None):
            out = await self.aclient.chat.completions.create(**self._request(prompt, schema))
            text = out.choices[0].message.content
            return schema.model_validate_json(text) if schema else text

    return OpenAICompatibleJudge


def shared_metric_class():
    from deepeval.metrics import SummarizationMetric

    class SharedSummarization(SummarizationMetric):
        """SummarizationMetric whose document-side judge outputs (truths,
        assessment questions, the document's answers) come from a per-document
        cache shared by every summary of that document (see module docstring)."""

        def __init__(self, cache, locks, **kw):
            super().__init__(**kw)
            self._cache, self._locks, self._doc = cache, locks, None

        async def _shared(self, text, key, make):
            k = hashlib.sha1(text.encode()).hexdigest()
            entry = self._cache.setdefault(k, {})
            if key not in entry:
                async with self._locks[(k, key)]:
                    if key not in entry:
                        entry[key] = await make()
            return entry[key]

        async def a_measure(self, test_case, *a, **kw):
            self._doc = test_case.input
            return await super().a_measure(test_case, *a, **kw)

        async def _a_generate_truths(self, text):
            make = super()._a_generate_truths
            return await self._shared(text, "truths", lambda: make(text))

        async def _a_generate_assessment_questions(self, text):
            make = super()._a_generate_assessment_questions
            return await self._shared(text, "questions", lambda: make(text))

        async def _a_generate_answers(self, text):
            make = super()._a_generate_answers
            if text != self._doc:
                return await make(text)
            return await self._shared(text, "answers", lambda: make(text))

    # deepeval finds its prompt templates by self.__class__.__name__, so the
    # subclass must carry the parent's name or the verdict steps find none.
    SharedSummarization.__name__ = SharedSummarization.__qualname__ = "SummarizationMetric"
    return SharedSummarization


async def run_deepeval(rows, path, judge, cache, locks, args):
    from deepeval.test_case import LLMTestCase

    Metric = shared_metric_class()
    done = load_done(path)
    todo = [r for r in rows if r[0] not in done]
    sem = asyncio.Semaphore(args.concurrency)
    errors = 0

    async def one(t, fh):
        nonlocal errors
        async with sem:
            m = Metric(cache, locks, threshold=0.5, model=judge, n=args.questions, include_reason=False,
                       async_mode=True, truths_extraction_limit=args.truths_limit)
            try:
                await m.a_measure(LLMTestCase(input=t[1], actual_output=t[3]), _show_indicator=False)
                rec = {"id": t[0], "deepeval": m.score, "alignment": m.score_breakdown["Alignment"],
                       "coverage": m.score_breakdown["Coverage"], "claims": len(m.claims)}
            except Exception as e:  # a judge failure on one sample must not lose the rest
                errors += 1
                rec = {"id": t[0], "error": f"{type(e).__name__}: {e}"[:300]}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if rec.get("error") is None:
                done[t[0]] = rec

    with path.open("a") as fh:
        await asyncio.gather(*(one(t, fh) for t in todo))
    if errors:
        print(f"  {errors} samples failed in the judge; re-run to retry them", file=sys.stderr)
    return [done[t[0]] for t in rows if t[0] in done]


async def run_judge(name, log, rows, path, out_log, args):
    """Re-score the log's samples with the Inspect LLM judge (summ_judge.py)."""
    from inspect_ai import score_async
    from inspect_ai.log import write_eval_log
    from inspect_ai.model import GenerateConfig, get_model

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from summ_judge import DIMENSIONS, summ_judge

    done = load_done(path)
    docs = {r[0]: r[1] for r in rows}
    todo = [s for s in log.samples if str(s.id) in docs and str(s.id) not in done]
    if todo:
        for s in todo:  # cnndm keeps only a row index; give the judge the document itself
            s.metadata = {**(s.metadata or {}), "document": docs[str(s.id)]}
        part = log.model_copy(update={"samples": todo})
        grader = get_model(f"openai-api/local/{args.judge_model}", base_url=args.judge_url, api_key="none",
                           config=GenerateConfig(max_connections=args.concurrency))
        # model= as well as the role: left unset, score_async instantiates the
        # log's own model (the evaluated 4B), which scoring never calls.
        scored = await score_async(part, [summ_judge(task=args.task, with_reference=args.judge_reference,
                                                     max_tokens=args.judge_max_tokens)],
                                   model=grader, model_roles={"grader": grader}, action="append", display="none")
        out_log.parent.mkdir(parents=True, exist_ok=True)
        # A retry scores only the samples still missing; keep the earlier judged
        # log and write the retry beside it rather than over it.
        n = 0
        while out_log.exists():
            n += 1
            out_log = out_log.with_name(out_log.name.removesuffix(".eval").split(".retry")[0] + f".retry{n}.eval")
        write_eval_log(scored, str(out_log))
        with path.open("a") as fh:
            for s in scored.samples:
                sc = next((v for k, v in (s.scores or {}).items() if "summ_judge" in k), None)
                value = sc.value if sc and isinstance(sc.value, dict) else {}
                rec = {"id": str(s.id), **{d: value.get(d) for d in (*DIMENSIONS, "judge_mean")}}
                if rec["judge_mean"] is None or rec["judge_mean"] != rec["judge_mean"]:
                    rec = {"id": str(s.id), "error": (sc.explanation if sc else "no judge score")[:300]}
                fh.write(json.dumps(rec) + "\n")
                if rec.get("error") is None:
                    done[rec["id"]] = rec
        failed = sum(1 for s in todo if str(s.id) not in done)
        if failed:
            print(f"  {failed} samples got no usable judgement; re-run to retry them", file=sys.stderr)
    return [done[i] for i in docs if i in done]


def run_summeval(rows, path):
    """SummEval's toolkit metrics (summ_extra.py) and BERTScore in the toolkit's
    default configuration (bert-base-uncased, layer 8, no baseline rescaling).
    Cheap and deterministic, so recomputed in full each time."""
    from bert_score import BERTScorer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from summ_extra import summeval_scores

    preds, refs, docs = [r[3] for r in rows], [r[2] for r in rows], [r[1] for r in rows]
    per, corpus = summeval_scores(preds, refs, docs)
    scorer = run_summeval.scorer = getattr(run_summeval, "scorer", None) or BERTScorer(
        model_type="bert-base-uncased", num_layers=8, lang="en", rescale_with_baseline=False)
    p, r, f = scorer.score(preds, refs, batch_size=32)
    with path.open("w") as fh:
        for row, rec, pi, ri, fi in zip(rows, per, p.tolist(), r.tolist(), f.tolist()):
            rec.update(id=row[0], bertscore_se_p=pi, bertscore_se_r=ri, bertscore_se_f1=fi)
            fh.write(json.dumps(rec) + "\n")
    keys = sorted({k for rec in per for k in rec} - {"id"})
    return {k: mean(per, k) for k in keys} | corpus


def mean(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log_dir", nargs="?", default="logs")
    ap.add_argument("--task", required=True, choices=["xsum", "billsum", "billsum_split", "cnndm", "samsum", "summeval"])
    ap.add_argument("--models", help="regex on the model label (adapter or merged directory, or base-...)")
    ap.add_argument("--size", choices=["4b", "1b", "27b"], default="4b")
    ap.add_argument("--prompt-variant", default="short", help="only logs run with this prompt_variant ('' for any)")
    ap.add_argument("--dataset", help="only logs whose dataset_path contains this, e.g. xsum_screen/test")
    ap.add_argument("--metrics", nargs="+", choices=["bertscore", "deepeval", "judge", "summeval"],
                    default=["bertscore", "deepeval", "judge"])
    ap.add_argument("--limit", type=int, help="first N samples of each log (for a quick check)")
    ap.add_argument("--out", type=Path, default=Path("../results/summ_metrics"))
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--judge-model", default="judge", help="served model name at --judge-url")
    ap.add_argument("--judge-label", help="name for the judge in file names (default: --judge-model)")
    ap.add_argument("--judge-max-tokens", type=int, default=4096)
    ap.add_argument("--questions", type=int, default=5, help="DeepEval's n: assessment questions per document")
    ap.add_argument("--truths-limit", type=int, help="DeepEval's truths_extraction_limit (default: none)")
    ap.add_argument("--concurrency", type=int, default=32, help="summaries judged at once")
    ap.add_argument("--judge-reference", action="store_true",
                    help="show the Inspect judge the reference as a style example (summ_judge.py)")
    args = ap.parse_args()

    out = args.out / args.task
    out.mkdir(parents=True, exist_ok=True)
    logs = latest_logs(args.log_dir, args.task, args.models, args.size, args.prompt_variant, args.dataset)
    if not logs:
        raise SystemExit(f"no successful {args.task} logs match {args.models!r}")
    jl = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.judge_label or args.judge_model)
    de_tag = f"deepeval-{jl}-n{args.questions}" + (f"-t{args.truths_limit}" if args.truths_limit else "")
    bs_tag = "bertscore-" + re.sub(r"[^A-Za-z0-9_.-]+", "_", args.bertscore_model)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from summ_judge import VERSION as JUDGE_VERSION
    jg_tag = f"judge{JUDGE_VERSION}-{jl}" + ("-ref" if args.judge_reference else "")

    judge = cache = None
    cache_path = out / f"judge-cache-{de_tag}.json"
    if "deepeval" in args.metrics:
        judge = judge_class()(args.judge_url, args.judge_model, args.judge_max_tokens)
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    async def score_logs():
        # One event loop for every log: the judge's async client and the cache
        # locks are bound to the loop they first run in.
        locks = collections.defaultdict(asyncio.Lock)
        for name, log in logs.items():
            rows = rows_of(log, args.task, args.limit)
            entry = summary.setdefault(name, {})
            entry.update(log=Path(log.location).name, samples=len(rows))
            if "bertscore" in args.metrics:
                recs = run_bertscore(rows, out / f"{name}.{bs_tag}.jsonl", args.bertscore_model)
                entry[bs_tag] = {k: mean(recs, k) for k in ("bertscore_p", "bertscore_r", "bertscore_f1")}
            if "deepeval" in args.metrics:
                recs = await run_deepeval(rows, out / f"{name}.{de_tag}.jsonl", judge, cache, locks, args)
                entry[de_tag] = {k: mean(recs, k) for k in ("deepeval", "alignment", "coverage")} | {"scored": len(recs)}
                cache_path.write_text(json.dumps(cache))
            if "summeval" in args.metrics:
                entry["summeval-toolkit"] = run_summeval(rows, out / f"{name}.summeval-toolkit.jsonl")
            if "judge" in args.metrics:
                recs = await run_judge(name, log, rows, out / f"{name}.{jg_tag}.jsonl",
                                       out / "judged" / f"{name}.{jg_tag}.eval", args)
                entry[jg_tag] = {k: mean(recs, k) for k in
                                 ("coherence", "consistency", "fluency", "relevance", "judge_mean")} | {"scored": len(recs)}
            # Re-read before writing: several jobs (one per group of models) may
            # share this file, and each must only replace its own models' entries.
            disk = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            disk[name] = entry
            summary_path.write_text(json.dumps(disk, indent=2) + "\n")
            print(name, json.dumps({k: v for k, v in entry.items() if k.startswith(("bertscore", "deepeval", "judge", "summeval"))}), flush=True)

    asyncio.run(score_logs())

    cols = [(bs_tag, "bertscore_f1", "BERTScore F1")] * ("bertscore" in args.metrics) + \
           [(de_tag, k, h) for k, h in (("deepeval", "DeepEval"), ("alignment", "align"), ("coverage", "cover"))
            if "deepeval" in args.metrics] + \
           [(jg_tag, k, h) for k, h in (("coherence", "coher"), ("consistency", "consist"), ("fluency", "fluency"),
                                         ("relevance", "relev")) if "judge" in args.metrics]
    print(f"\n{'model':44} {'n':>5} " + " ".join(f"{h:>12}" for _, _, h in cols))
    for name in logs:
        e = summary[name]
        cells = [e.get(tag, {}).get(k) for tag, k, _ in cols]
        print(f"{name:44} {e['samples']:5d} " + " ".join(f"{c:12.4f}" if c is not None else f"{'-':>12}" for c in cells))


if __name__ == "__main__":
    main()
