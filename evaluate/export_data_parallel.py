# Exports the data-parallel evaluations from one base model as JSON
# (eval_to_json.py), one file per model, laid out as export_results.py lays out
# the 4B crime and XSum ones:
#
#   <out>/<task>_data_parallel_gemma/  base.json  index.tsv
#     seed_<S>/  full.json
#       split_<N>/  <i>_of_<N>.json  merge_<method>.json
#
# Usage, from evaluate/:
#   uv run python export_data_parallel.py --base google/gemma-3-1b-it --logs logs-1b \
#       --out ../results/gemma3_1b crime wine xsum mt_domain_de_en [--workers N] [--dry-run]
#
# Logs are matched to models by make_data_parallel_tables.py's latest(), so each
# file is the evaluation its tables report: the latest successful log of that
# model. XSum's and CLINC150's prompts are kept once, in
# <out>/<shared prompts file>, which each of their JSONs names. A file that exists
# and was made from the chosen log is skipped, so a rerun resumes; one made from
# another log (the model was evaluated again) is converted again. index.tsv maps
# every file to its log.
import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from eval_to_json import convert, make_prompts
from make_data_parallel_tables import BASE, METHODS, TASKS, latest, spec_for

# Prompts the same in every eval of a task are kept once: XSum's articles, and
# CLINC150's prompt, which lists all 150 labels.
SHARED_PROMPTS = {"xsum": "xsum_prompts_long.json", "clinc150": "clinc150_prompts_labels.json"}


def target(spec, seed, name):
    """The file (relative to the task's folder) for one model, or None."""
    if seed is None:
        return "base.json"
    if name == spec["full"]:
        return f"seed_{seed}/full.json"
    if m := re.fullmatch(spec["part"], name):
        return f"seed_{seed}/split_{m[2]}/{m[1]}_of_{m[2]}.json"
    if (m := re.fullmatch(spec["merge"], name)) and m[2] in METHODS:
        return f"seed_{seed}/split_{m[1]}/merge_{m[2]}.json"
    return None


def source_log(path):
    """The source_log of an exported JSON: its header's first key, so in its first few lines."""
    with open(path) as f:
        m = re.search(r'"source_log": "([^"]+)"', f.read(4096))
    return m[1] if m else None


def job(log, out, prompts):
    try:
        return convert(log, out, prompts), None
    except BaseException as e:  # convert() exits on an unfinished log
        return None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+", choices=TASKS)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--logs", default="logs", help="the directory of .eval logs (default %(default)s)")
    ap.add_argument("--out", required=True, help="the results folder to write the task folders into")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    todo, index, failed = [], {}, []
    for task in args.tasks:
        spec, folder, _ = spec_for(task, args.base)
        found = latest(task, args.base, folder, args.logs)
        d = out / f"{task}_data_parallel_gemma"
        rows = []
        for (seed, name), (_, _, log) in sorted(found.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
            t = target(spec, seed, name)
            if t is None:
                print(f"  no file for {task} seed {seed} {name}", file=sys.stderr)
                continue
            rows.append((t, Path(log).name))
            if not (d / t).exists() or source_log(d / t) != Path(log).name:
                todo.append((task, str(log), d / t))
        index[d] = rows
        print(f"{task}: {len(rows)} models, {sum(1 for x in todo if x[0] == task)} to convert")
        if task in SHARED_PROMPTS and not (out / SHARED_PROMPTS[task]).exists() and not args.dry_run:
            base_log = found.get((None, "base"))
            log = base_log[2] if base_log else next(iter(found.values()))[2]
            print(f"{out / SHARED_PROMPTS[task]}: {make_prompts(log, out / SHARED_PROMPTS[task])} prompts")
    if args.dry_run:
        return

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futures = {ex.submit(job, log, str(path),
                             str(out / SHARED_PROMPTS[task]) if task in SHARED_PROMPTS else None): path
                   for task, log, path in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            n, err = fut.result()
            if err:
                failed.append(futures[fut])
                print(f"FAILED {futures[fut]}: {err}", file=sys.stderr)
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    for d, rows in index.items():
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.tsv").write_text("file\tsource_log\n" + "".join(f"{a}\t{b}\n" for a, b in rows if (d / a).exists()))
    if failed:
        sys.exit(f"{len(failed)} failed (rerun to retry)")


if __name__ == "__main__":
    main()
