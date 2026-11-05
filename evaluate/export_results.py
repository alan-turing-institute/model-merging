# Exports the evaluations of the merging studies from logs/ to ../results/ as
# JSON (eval_to_json.py), one file per model and task, laid out like models/:
#
#   crime_data_parallel_gemma/  base.json
#     seed_<S>/  full.json
#       split_<N>/  <i>_of_<N>.json  merge_<method>.json
#   xsum_data_parallel_gemma/   the same (gold targets, long prompt)
#   task_merging/<gemma|mistral|llama>/  base_<task>.json
#     seed_<S>/  expert_<task>.json  <method>/<task>.json  moe/<task>.json
#
# Usage, from evaluate/:
#   uv run python export_results.py [folder ...] [--workers N] [--dry-run]
# with folder one or more of crime_data_parallel_gemma, xsum_data_parallel_gemma,
# task_merging (default: all three).
#
# A log is matched to its file through the model_provenance path that
# evaluate.sbatch records in its metadata (the base models by their Hub id);
# CLINC150 counts only with the labels prompt and XSum with the long one. Where
# a model was evaluated more than once, the latest successful log counts, as in
# make_data_parallel_tables.py. Files that exist are skipped, so a rerun resumes.
# XSum's prompts are kept once, in xsum_prompts_long.json; each XSum JSON gives
# its path as prompts_file and holds inline only a prompt that differs from it.
# Each folder gets index.tsv: every file and the .eval it came from. Conversions
# run in a process pool, largest logs first; ~15 s of each is loading Inspect,
# which the pool pays once per worker.
import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from inspect_ai.log import read_eval_log

from eval_to_json import convert, make_prompts

LOGS = Path("logs")
RESULTS = Path("../results")
# XSum's prompts (whole articles) are most of each file and the same in every
# eval, so they are kept once, here, and each XSum JSON refers to it; a prompt
# that differs stays inline (see eval_to_json.py --prompts).
SHARED_PROMPTS = {"xsum": RESULTS / "xsum_prompts_long.json"}
FOLDERS = ["crime_data_parallel_gemma", "xsum_data_parallel_gemma", "task_merging"]
TASKS = {"crime", "wine", "clinc150", "xsum"}
PROMPT = {"clinc150": "labels", "xsum": "long"}
METHODS = "linear|ties|dare_ties|task_arithmetic|model_stock|slerp|arcee_fusion"
STEMS = {"gemma3_4b": "gemma", "mistral_7b": "mistral", "llama3_8b": "llama"}
BASES = {"hf/google/gemma-3-4b-it": "gemma", "hf/mistralai/Mistral-7B-Instruct-v0.3": "mistral",
         "hf/meta-llama/Llama-3.1-8B-Instruct": "llama"}


def header(path):
    h = read_eval_log(str(path), header_only=True)
    md = h.eval.metadata or {}
    return {"path": path, "size": path.stat().st_size, "status": h.status, "task": h.eval.task,
            "created": h.eval.created, "prompt": (h.eval.task_args or {}).get("prompt_variant"),
            "model": h.eval.model, "model_args": set(h.eval.model_args or {}),
            "kind": md.get("model_kind"), "prov": md.get("model_provenance") or ""}


def targets(h):
    """The result files (relative to results/) a log is for: usually one, two for
    a base Gemma crime/XSum eval (data parallel and task merging)."""
    t, prov = h["task"], h["prov"]
    if t not in TASKS or (t in PROMPT and h["prompt"] != PROMPT[t]):
        return []
    # The base model alone: a Hub id, no adapter or local path. Logs from before
    # evaluate.sbatch recorded model_kind have no metadata.
    if h["model"] in BASES and h["kind"] in ("base", None) and not {"adapter", "model_path"} & h["model_args"]:
        out = [f"task_merging/{BASES[h['model']]}/base_{t}.json"]
        if h["model"] == "hf/google/gemma-3-4b-it" and t in ("crime", "xsum"):
            out.append(f"{t}_data_parallel_gemma/base.json")
        return out
    # Data parallel (Gemma; crime, and XSum's gold-target long-prompt students).
    if t in ("crime", "xsum"):
        tag, suf = ("gemma3_4b_crime", "") if t == "crime" else ("gemma3_4b_xsum_gold", "_long")
        d = f"{t}_data_parallel_gemma"
        if m := re.search(rf"models/{t}/seed_(\d)/(?:split_\d/)?([^/]+)/provenance\.json$", prov):
            s, name = m.groups()
            if name == tag + suf:
                return [f"{d}/seed_{s}/full.json"]
            if mm := re.fullmatch(rf"{tag}_(\d)_of_(\d){suf}", name):
                return [f"{d}/seed_{s}/split_{mm[2]}/{mm[1]}_of_{mm[2]}.json"]
            if mm := re.fullmatch(rf"{tag}{suf}_merge_(\d)_({METHODS})", name):
                return [f"{d}/seed_{s}/split_{mm[1]}/merge_{mm[2]}.json"]
    # Task merging: the MLP-only experts, on their own task...
    if m := re.search(r"models/([a-z0-9]+)/seed_(\d)/([^/]+_mlp)/provenance\.json$", prov):
        task_dir, s, name = m.groups()
        stem = next((k for k in STEMS if name.startswith(k + "_")), None)
        if stem and task_dir == t:
            return [f"task_merging/{STEMS[stem]}/seed_{s}/expert_{t}.json"]
    # ...and the dense merges and the MoE, on all four.
    if m := re.search(rf"models/task_merge/seed_(\d)/([a-z0-9_]+?)_task_(?:merge_4_({METHODS})|(moe)_4)/provenance\.json$", prov):
        s, stem, method, moe = m.groups()
        if stem in STEMS:
            return [f"task_merging/{STEMS[stem]}/seed_{s}/{method or moe}/{t}.json"]
    return []


def select(folders, workers):
    """{result file: header of its log}, the latest successful log for each."""
    with ProcessPoolExecutor(workers) as ex:
        heads = list(ex.map(header, sorted(LOGS.glob("*.eval")), chunksize=8))
    best = {}
    for h in heads:
        if h["status"] != "success":
            continue
        for target in targets(h):
            if target.split("/")[0] in folders and (target not in best or h["created"] > best[target]["created"]):
                best[target] = h
    return best


def job(log, out, prompts):
    try:
        return convert(log, out, prompts), None
    except BaseException as e:  # convert() exits on an unfinished log
        return None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*", help=f"any of {', '.join(FOLDERS)} (default: all)")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    folders = args.folders or FOLDERS
    if bad := set(folders) - set(FOLDERS):
        ap.error(f"unknown folder(s) {sorted(bad)}: choose from {FOLDERS}")

    best = select(set(folders), args.workers)
    for f in folders:
        n = sum(1 for t in best if t.startswith(f + "/"))
        gb = sum(h["size"] for t, h in best.items() if t.startswith(f + "/")) / 1e9
        print(f"{f}: {n} files from {gb:.2f} GB of logs")
    todo = sorted(((t, h) for t, h in best.items() if not (RESULTS / t).exists()),
                  key=lambda th: -th[1]["size"])
    print(f"{len(best) - len(todo)} exist already, {len(todo)} to convert")
    if args.dry_run:
        for t, h in todo:
            print(f"  {t}  <-  {h['path'].name}")
        return

    # The shared prompt files, from the base model's log (any log of the task
    # would do: prompts depend only on the task and its prompt variant).
    for task, path in SHARED_PROMPTS.items():
        logs = sorted((h for t, h in best.items() if h["task"] == task), key=lambda h: h["kind"] != "base")
        if logs and not path.exists():
            print(f"{path}: {make_prompts(str(logs[0]['path']), path)} prompts from {logs[0]['path'].name}")

    failed = []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futures = {ex.submit(job, str(h["path"]), str(RESULTS / t),
                             str(SHARED_PROMPTS[h["task"]]) if h["task"] in SHARED_PROMPTS else None): t
                   for t, h in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            n, err = fut.result()
            if err:
                failed.append((futures[fut], err))
                print(f"FAILED {futures[fut]}: {err}", file=sys.stderr)
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)

    for f in folders:
        rows = sorted((t[len(f) + 1:], h["path"].name) for t, h in best.items()
                      if t.startswith(f + "/") and (RESULTS / t).exists())
        (RESULTS / f).mkdir(parents=True, exist_ok=True)
        (RESULTS / f / "index.tsv").write_text("file\tsource_log\n" + "".join(f"{a}\t{b}\n" for a, b in rows))
    if failed:
        sys.exit(f"{len(failed)} failed (rerun to retry): " + ", ".join(t for t, _ in failed))


if __name__ == "__main__":
    main()
