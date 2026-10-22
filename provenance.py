# Writes and reads the provenance.json kept next to each dataset (saved with
# save_to_disk) and each model: which script made it, from what, with which
# arguments, at which git commit and in which Slurm job.
#
# "coverage" says how much of the task's prepared data a dataset holds, so a
# model can be checked for full or partial data from one field:
#   "full"    - a prepare_data_*.py output
#   "i of n"  - a split_dataset.py piece
# A dataset built from another (generate_teacher_logprobs.py) takes its source's
# coverage and keeps the source's whole record under "source_provenance".
#
# "content_hash" fingerprints each split's rows, so two datasets (e.g. two
# "1 of 2" pieces) can be checked for holding the same data whatever their names
# or records say. It ignores row order: the same rows shuffled differently hash
# the same.
#
# Models get a record too, written after training or merging (see the CLI at
# the bottom). A trained model's record names its base model, method, the key
# hyperparameters, its final train/eval loss and its train/validation datasets
# with their whole records; its coverage is its training data's. A merged
# model's names the merge method and each input model with its weight and
# whole record; its coverage lists the inputs'. The full configs stay where
# Axolotl and mergekit put them (config.yaml, mergekit_config.yml). Weights are
# not hashed: the configs and data hashes already say how a model was made.
#
# describe() flattens a model's record into the --metadata evaluate.sbatch
# passes to Inspect, so every .eval log says what its model is.
#
# The scripts that use this live in train/, self-distill/, merge/, evaluate/
# and the repo root, each with its own environment; those in subdirectories put
# the root on sys.path before importing it. It imports only the standard
# library at the top (yaml when reading a model's config), since the merge
# environment has no `datasets`.
import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FILENAME = "provenance.json"
REPO = Path(__file__).resolve().parent

# Config keys copied into a trained model's record (when the config sets them):
# the ones that tell otherwise-identical runs apart.
HYPERPARAMETERS = [
    "adapter", "load_in_4bit", "lora_r", "lora_alpha", "num_epochs", "learning_rate",
    "micro_batch_size", "gradient_accumulation_steps", "sequence_len",
    "kd_alpha", "kd_ce_alpha", "kd_temperature",
]


def git_state():
    """(commit, dirty) of this repo, or (None, None) outside git."""
    def git(*cmd):
        return subprocess.run(
            ["git", *cmd], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def rows(ds):
    """Row count per split, or {"all": n} for a plain Dataset."""
    if isinstance(ds, dict):  # a DatasetDict
        return {split: len(ds[split]) for split in ds}
    return {"all": len(ds)}


def content_hash(ds):
    """Per split, a 16-hex-digit hash of the set of rows, independent of their order."""
    def split_hash(split):
        row_hashes = sorted(
            hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).digest()
            for row in split.with_format(None)
        )
        return hashlib.sha256(b"".join(row_hashes)).hexdigest()[:16]
    if isinstance(ds, dict):
        return {split: split_hash(ds[split]) for split in ds}
    return {"all": split_hash(ds)}


def read(path):
    """The provenance record in directory path, or None if it has none."""
    path = Path(path) / FILENAME
    return json.loads(path.read_text()) if path.exists() else None


def _jsonable(value):
    """value with NaN/inf floats as strings, which strict JSON readers reject as numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _save(directory, record):
    (Path(directory) / FILENAME).write_text(json.dumps(_jsonable(record), indent=2) + "\n")


def _run_fields():
    commit, dirty = git_state()
    return {"git_commit": commit, "git_dirty": dirty, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(dataset_dir, script, coverage, ds, args=None, **fields):
    """Write dataset_dir/provenance.json for a dataset.

    script is the caller's __file__; args its argparse namespace (Paths are
    stored as strings); fields any further script-specific entries.
    """
    _save(dataset_dir, {
        "script": str(Path(script).resolve().relative_to(REPO)),
        "created": _now(),
        "coverage": coverage,
        **fields,
        "rows": rows(ds),
        "content_hash": content_hash(ds),
        "args": (
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            if args is not None else None
        ),
        **_run_fields(),
    })


# --- Models -----------------------------------------------------------------

def resolve(path, base_dir):
    """path made absolute against base_dir, textually: models/ and datasets/ are
    symlinks, so following ".." physically would leave the repo."""
    return os.path.normpath(os.path.join(os.path.abspath(base_dir), path))


def final_losses(model_dir):
    """The last logged train and eval loss, from the latest checkpoint's trainer state."""
    states = glob.glob(os.path.join(model_dir, "checkpoint-*", "trainer_state.json"))
    if not states:
        return None
    latest = max(states, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    state = json.loads(Path(latest).read_text())
    train = [e for e in state["log_history"] if "loss" in e]
    evals = [e for e in state["log_history"] if "eval_loss" in e]
    return {
        "final_train_loss": train[-1]["loss"] if train else None,
        "final_train_step": train[-1]["step"] if train else None,
        "final_eval_loss": evals[-1]["eval_loss"] if evals else None,
        "final_eval_step": evals[-1]["step"] if evals else None,
        "max_steps": state.get("max_steps"),
        "source": os.path.relpath(latest, model_dir),
    }


def _dataset_entry(path, base_dir):
    """A config's dataset path (<dataset>/<split>) as the dataset, split and record."""
    split_dir = resolve(path, base_dir)
    dataset_dir = os.path.dirname(split_dir)
    return {"path": dataset_dir, "split": os.path.basename(split_dir), "provenance": read(dataset_dir)}


def _coverage(entries):
    """One coverage if all entries agree, else a list (None where there is no record)."""
    found = [e["provenance"]["coverage"] if e["provenance"] else "unknown" for e in entries]
    return found[0] if len(set(found)) == 1 else found


def trained_record(model_dir, base_dir):
    """What made an Axolotl output dir, from its config.yaml and checkpoints."""
    import yaml
    cfg = yaml.safe_load(Path(model_dir, "config.yaml").read_text())
    train = [_dataset_entry(d["path"], base_dir) for d in cfg.get("datasets", [])]
    validation = [_dataset_entry(d["path"], base_dir) for d in cfg.get("test_datasets") or []]
    return {
        "coverage": _coverage(train),
        "kind": "trained",
        "method": "kd" if cfg.get("kd_trainer") else "sft",
        "base_model": cfg["base_model"],
        "config": "config.yaml",
        "hyperparameters": {k: cfg[k] for k in HYPERPARAMETERS if k in cfg},
        "losses": final_losses(model_dir),
        "train_datasets": train,
        "validation_datasets": validation,
    }


def _model_entry(spec, base_dir):
    """A model named in a merge config: a local dir (with its record) or a Hub id."""
    local = resolve(spec, base_dir)
    if os.path.isdir(local):
        return {"path": local, "provenance": read(local)}
    return {"hub_id": spec, "provenance": None}


def merged_record(model_dir, base_dir):
    """What made a mergekit output dir, from its mergekit_config.yml."""
    import yaml
    cfg = yaml.safe_load(Path(model_dir, "mergekit_config.yml").read_text())
    inputs = []
    for m in cfg.get("models", []):
        # "<base>+<adapter>" is mergekit's syntax for a LoRA adapter on a base.
        base, _, adapter = m["model"].partition("+")
        entry = _model_entry(adapter or base, base_dir)
        if adapter:
            entry["on_base_model"] = base
        inputs.append({**entry, "parameters": m.get("parameters")})
    def coverage(i):
        if i["provenance"]:
            return i["provenance"]["coverage"]
        return "base" if "hub_id" in i else "unknown"
    return {
        "coverage": [coverage(i) for i in inputs],
        "kind": "merged",
        "merge_method": cfg.get("merge_method"),
        "base_model": cfg.get("base_model"),
        "config": "mergekit_config.yml",
        "parameters": cfg.get("parameters"),
        "inputs": inputs,
    }


def write_model(model_dir, script, base_dir=".", **overrides):
    """Write model_dir/provenance.json after training (config.yaml) or merging
    (mergekit_config.yml). Relative paths in the config resolve against base_dir,
    the directory the job ran from. overrides replace fields (for backfills)."""
    merged = Path(model_dir, "mergekit_config.yml").exists()
    body = merged_record(model_dir, base_dir) if merged else trained_record(model_dir, base_dir)
    record = {
        "script": str(Path(script).resolve().relative_to(REPO)),
        "created": _now(),
        **body,
        **_run_fields(),
    }
    record.update(overrides)
    _save(model_dir, record)


def describe(model):
    """A model's record flattened for Inspect's --metadata. model is a model or
    adapter dir, or a Hugging Face Hub id (then a base model with no record)."""
    record = read(model) if os.path.isdir(model) else None
    if record is None:
        return {"model_kind": "base" if not os.path.isdir(model) else "unknown"}
    meta = {
        "model_kind": record["kind"],
        "model_provenance": os.path.join(os.path.abspath(model), FILENAME),
        "model_coverage": record["coverage"],
        "model_git_commit": record.get("git_commit"),
        "model_slurm_job": record.get("slurm_job_id"),
    }
    if record["kind"] == "trained":
        meta.update({
            "method": record["method"],
            "base_model": record["base_model"],
            **record["hyperparameters"],
            **{k: v for k, v in (record["losses"] or {}).items() if k.startswith("final_")},
        })
        train = record["train_datasets"][0] if record["train_datasets"] else None
        if train:
            meta["train_data"] = os.path.basename(train["path"])
            data = train["provenance"] or {}
            meta["train_data_hash"] = (data.get("content_hash") or {}).get(train["split"])
            if "teacher_model" in data:
                meta["teacher_model"] = data["teacher_model"]
                meta["targets"] = data["targets"]
    else:
        meta.update({
            "merge_method": record["merge_method"],
            "merged_models": [os.path.basename(i.get("path") or i["hub_id"]) for i in record["inputs"]],
            "merge_weights": [(i["parameters"] or {}).get("weight") for i in record["inputs"]],
        })
    return meta


def main():
    parser = argparse.ArgumentParser(description="Write or read model provenance records.")
    sub = parser.add_subparsers(dest="command", required=True)
    model = sub.add_parser("model", help="Write <model_dir>/provenance.json after training or merging.")
    model.add_argument("model_dir", help="An Axolotl or mergekit output directory.")
    model.add_argument("--script", required=True, help="The script that ran, e.g. train.sbatch.")
    model.add_argument(
        "--base-dir", default=".",
        help="Directory relative paths in the config resolve against (default: the current one).",
    )
    desc = sub.add_parser(
        "describe",
        help="Print a model's record as key=<json> lines, one per Inspect --metadata argument.",
    )
    desc.add_argument("model", help="A model or adapter dir, or a Hugging Face Hub id.")
    args = parser.parse_args()

    if args.command == "model":
        write_model(args.model_dir, args.script, args.base_dir)
        print(f"wrote {Path(args.model_dir) / FILENAME}")
    else:
        # JSON values survive Inspect's YAML parsing of --metadata intact (a
        # bare string would be split at commas).
        for key, value in describe(args.model).items():
            print(f"{key}={json.dumps(_jsonable(value))}")


if __name__ == "__main__":
    sys.exit(main())
