# Writes and reads the provenance.json kept next to each dataset saved with
# save_to_disk: which script made it, from what, with which arguments, at which
# git commit and in which Slurm job.
#
# "coverage" says how much of the task's prepared data a dataset holds, so a
# model (via the dataset its config.yaml names) can be checked for full or
# partial data from one field:
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
# The scripts that use this live in train/, self-distill/ and the repo root;
# those in subdirectories put the root on sys.path before importing it.
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from datasets import DatasetDict

FILENAME = "provenance.json"
REPO = Path(__file__).resolve().parent


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
    if isinstance(ds, DatasetDict):
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
    if isinstance(ds, DatasetDict):
        return {split: split_hash(ds[split]) for split in ds}
    return {"all": split_hash(ds)}


def read(dataset_dir):
    """A dataset's provenance record, or None if it has none."""
    path = Path(dataset_dir) / FILENAME
    return json.loads(path.read_text()) if path.exists() else None


def write(dataset_dir, script, coverage, ds, args=None, **fields):
    """Write dataset_dir/provenance.json.

    script is the caller's __file__; args its argparse namespace (Paths are
    stored as strings); fields any further script-specific entries.
    """
    commit, dirty = git_state()
    record = {
        "script": str(Path(script).resolve().relative_to(REPO)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage": coverage,
        **fields,
        "rows": rows(ds),
        "content_hash": content_hash(ds),
        "args": (
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            if args is not None else None
        ),
        "git_commit": commit,
        "git_dirty": dirty,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (Path(dataset_dir) / FILENAME).write_text(json.dumps(record, indent=2) + "\n")
