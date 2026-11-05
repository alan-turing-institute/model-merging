# Writes a compact JSON record of one Inspect .eval log: what was evaluated and
# how, the aggregate metrics, and per sample the prompt, the model's reply, the
# target and the scores. It drops what the .eval repeats (each prompt is stored
# there ~4 times) and the event trace, so it is ~10x smaller on disk and ~50x
# smaller in git, which compresses its repetition (e.g. CLINC150's label list).
#
# Usage, from evaluate/:
#   uv run python eval_to_json.py <log.eval> <out.json> [--prompts <prompts.json>]
#   uv run python eval_to_json.py --make-prompts <log.eval> <prompts.json>
#
# --prompts: a shared prompt file (made by --make-prompts from any one log of the
# same task and prompt variant). A sample whose prompt is identical to the
# shared one for its id is written without it; any other keeps its prompt
# inline, so no prompt is ever lost. The header names the file ("prompts_file",
# relative to the output's results/ root where possible) and counts the samples
# that use it. For XSum, whose prompts (whole articles) are most of each file
# and the same in every eval.
#
# Only a successful, complete log is converted. NaN (e.g. a CLINC150 shard
# accuracy for a query outside that shard) is written as null, so the file is
# strict JSON. The header is indented; each sample is one line.
import argparse
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

from inspect_ai.log import read_eval_log


def clean(x):
    """NaN/inf -> None, recursively, so json.dumps gives strict JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def prompt_of(s):
    # messages is the conversation as run: the prompt (any system turn
    # included), then the model's reply. The prompt is everything before the
    # last assistant message.
    msgs = list(s.messages)
    last = max((i for i, m in enumerate(msgs) if m.role == "assistant"), default=len(msgs))
    return [{"role": m.role, "content": m.text} for m in msgs[:last]]


def sample_record(s, shared=None):
    prompt = prompt_of(s)
    rec = {"id": s.id}
    if shared is None or shared.get(str(s.id)) != prompt:
        rec["prompt"] = prompt
    rec.update({
        "output": s.output.completion,
        "target": s.target,
        "scores": {name: {"value": sc.value, "answer": sc.answer, "explanation": sc.explanation}
                   for name, sc in (s.scores or {}).items()},
    })
    return rec


def read_complete(log_path):
    log = read_eval_log(log_path)
    r = log.results
    if log.status != "success" or r is None or r.completed_samples != r.total_samples:
        sys.exit(f"{log_path}: status {log.status}, "
                 f"{r.completed_samples if r else 0}/{r.total_samples if r else '?'} samples: not converted")
    return log


def write_json(out_path, head, lines):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The header's closing brace is replaced by the samples list, one per line.
    # Written under a temporary name and renamed, so an interrupted run never
    # leaves a partial file that looks finished.
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(clean(head), indent=1, ensure_ascii=False, allow_nan=False, default=str)[:-1].rstrip()
                   + ',\n "samples": [\n' + ",\n".join(lines) + "\n ]\n}\n")
    tmp.replace(out)


@lru_cache
def load_prompts(path):
    return json.loads(Path(path).read_text())["prompts"]


def make_prompts(log_path, out_path):
    """The shared prompt file: each sample id's prompt, from one log."""
    log = read_complete(log_path)
    head = {"source_log": Path(log_path).name, "task": log.eval.task, "task_args": log.eval.task_args,
            "dataset": log.eval.dataset.name}
    # Written whole (prompts is a dict, not a list), so not through write_json.
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({**clean(head), "prompts": {str(s.id): prompt_of(s) for s in log.samples}},
                              indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(out)
    return len(log.samples)


def convert(log_path, out_path, prompts_path=None):
    log = read_complete(log_path)
    shared = load_prompts(str(prompts_path)) if prompts_path else None
    ev, r = log.eval, log.results
    header = {
        "source_log": Path(log_path).name,
        "eval_id": ev.eval_id,
        "created": ev.created,
        "task": ev.task,
        "task_args": ev.task_args,
        "dataset": ev.dataset.name,
        "model": ev.model,
        "model_args": ev.model_args,
        "generate_config": log.plan.config.model_dump(exclude_none=True),
        "metadata": ev.metadata,  # evaluate.sbatch's provenance description and slurm job
        "inspect_ai": ev.packages.get("inspect_ai"),
        "samples_total": r.total_samples,
        "metrics": {f"{sc.name}/{m.name}": m.value for sc in r.scores for m in sc.metrics.values()},
    }
    records = [sample_record(s, shared) for s in log.samples]
    if shared is not None:
        # The prompt file's path from the output's folder, e.g. ../../xsum_prompts_long.json.
        header["prompts_file"] = os.path.relpath(prompts_path, Path(out_path).parent)
        header["prompts_shared"] = sum("prompt" not in rec for rec in records)
    write_json(out_path, header, [json.dumps(clean(rec), ensure_ascii=False, allow_nan=False, default=str)
                                  for rec in records])
    return len(records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--prompts", help="shared prompt file: matching prompts are not repeated")
    ap.add_argument("--make-prompts", action="store_true", help="write the shared prompt file from <log> to <out>")
    a = ap.parse_args()
    if a.make_prompts:
        print(f"wrote {a.out}: {make_prompts(a.log, a.out)} prompts")
    else:
        print(f"wrote {a.out}: {convert(a.log, a.out, a.prompts)} samples")
