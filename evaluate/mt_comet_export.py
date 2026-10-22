"""Export (source, translation, reference) triples from the translation .eval
logs, for COMET scoring in the separate metrics environment.

    uv run python mt_comet_export.py logs ../results/mt_comet_inputs

COMET needs the English source as well as the output and reference, and the
logs hold the source only inside the rendered prompt. So it is read from the
test split each task scored - sample ids are row indices into it - and every
row's reference is checked against the log's target, so a split that has been
re-prepared since cannot be silently mis-paired.

One JSONL file per run: <task>_<size>_<base|trained>.jsonl.

With --all (for mt_comet_merges.py): every successful EN->DE run, the latest log
per model, named by its adapter or merged-model directory (or base-<model>) -
the experts, full and IID-half models, merges, task-parallel ceilings and
merges, distilled models - one <label>.jsonl each, with the sample's chrF++
alongside so the two metrics can be read together.
"""

import json
import sys
from pathlib import Path

from datasets import load_from_disk
from inspect_ai.log import list_eval_logs, read_eval_log

TASKS = {"mt_domain": "de", "mt_zh": "zh"}
REPO_ROOT = Path(__file__).resolve().parent.parent


def label(log):
    args = log.eval.model_args or {}
    for key in ("adapter_path", "model_path"):
        if args.get(key):
            return Path(str(args[key]).rstrip("/")).name
    return "base-" + str(log.eval.model).split("/")[-1]


def export_all(log_dir, out_dir, task="mt_domain"):
    """Every run on the task, latest log per label (headers to choose)."""
    infos = {}
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info, header_only=True)
        if log.status != "success" or log.eval.task.split("/")[-1] != task:
            continue
        name = label(log)
        if name not in infos or log.eval.created > infos[name][0].eval.created:
            infos[name] = (log, info)
    lang = TASKS[task]
    test = load_from_disk(str(REPO_ROOT / "datasets" / task / "test"))
    for name, (_, info) in sorted(infos.items()):
        log = read_eval_log(info)
        rows = []
        for sample in log.samples:
            row = test[int(sample.id)]
            if row[lang] != str(sample.target):
                raise SystemExit(f"{name} sample {sample.id}: log target does not match the test split")
            chrf = next((sc.value.get("chrf") for sc in (sample.scores or {}).values()
                         if isinstance(sc.value, dict) and "chrf" in sc.value), None)
            rows.append({"id": sample.id, "domain": row["domain"], "src": row["en"], "chrf": chrf,
                         "mt": sample.output.completion.strip() if sample.output else "", "ref": row[lang]})
        (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"wrote {len(rows)} triples for {name}")


def main():
    log_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if "--all" in sys.argv[3:]:
        out_dir.mkdir(parents=True, exist_ok=True)
        return export_all(log_dir, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = {}
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        task = log.eval.task.split("/")[-1]
        if log.status != "success" or task not in TASKS:
            continue
        size = "1b" if "gemma-3-1b" in log.eval.model else "4b" if "gemma-3-4b" in log.eval.model else None
        adapter = str((log.eval.model_args or {}).get("adapter_path", "")).rstrip("/")
        want = f"gemma3-{task}-full{'_1b' if size == '1b' else ''}-lora"
        kind = "base" if not adapter else "trained" if adapter.endswith(want) else None
        if size and kind:
            key = (task, size, kind)
            if key not in picked or log.eval.created > picked[key].eval.created:
                picked[key] = log

    for (task, size, kind), log in sorted(picked.items()):
        lang = TASKS[task]
        test = load_from_disk(str(REPO_ROOT / "datasets" / task / "test"))
        rows = []
        for sample in log.samples:
            row = test[int(sample.id)]
            if row[lang] != str(sample.target):
                raise SystemExit(f"{task} sample {sample.id}: log target does not match the test split")
            rows.append({"id": sample.id, "domain": row["domain"], "src": row["en"],
                         "mt": sample.output.completion.strip() if sample.output else "", "ref": row[lang]})
        path = out_dir / f"{task}_{size}_{kind}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"wrote {len(rows)} triples to {path.name}")


if __name__ == "__main__":
    main()
