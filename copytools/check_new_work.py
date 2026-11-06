#!/usr/bin/env python3
"""Compares what is on disk now with what yesterday's archives hold, to check
what has to be copied to Azure. Read-only: it writes nothing.

    cd $(realpath $PROJECTDIR/$USER/model-merging)/tars && python3 check_new_work.py

Every regular file under models/ and datasets/ (less the caches make_one.sh
excludes) is matched by path, size and mtime (to the minute, as tar lists it)
against every <name>.files.txt here:
  same     in an archive, unchanged             -> already on Azure
  changed  in an archive, but size/mtime differ  -> must be copied
  new      in no archive                         -> must be copied
  moved    under .../mismatched_length/..., identical to an archived file at
           the path without that component       -> already on Azure (old path)
  gone     in an archive, no longer on disk at that path (and not moved)
Files are grouped into units (one model, or one dataset directory), and the
units are checked against the expectation, by 6 Nov 2026:
  - the XSum models of seeds 1-5 were moved to mismatched_length/, then retrained;
  - task merging (dense merges and MoEs) was redone for seeds 1-3;
  - nothing else changed.
Last, the eval logs and results in the home checkout newer than plan.tsv
(none of which were archived) are counted.
"""
import argparse
import glob
import os
import re
import stat
import sys
import time
from collections import defaultdict

EXPECT_NEW = [r"^models/xsum/seed_[1-5]/", r"^models/task_merge/seed_[123]/"]
GROUP_DIR = re.compile(r"seed_\d+|split_\d+|mismatched_length")


def excluded(rel):
    return any(p == "intermediate" or p.startswith(".lora_merge_cache") for p in rel.split("/"))


def unit(rel):
    """models/xsum/seed_1/<model>/checkpoint-200/x -> models/xsum/seed_1/<model>;
    datasets/xsum/seed_1/train.jsonl -> datasets/xsum/seed_1."""
    dirs = rel.split("/")[:-1]
    hits = [i for i, p in enumerate(dirs) if GROUP_DIR.fullmatch(p)]
    if not hits:
        return "/".join(dirs[:3]) or rel
    return "/".join(dirs[:hits[-1] + 2])


def unmoved(rel):
    return "/".join(p for p in rel.split("/") if p != "mismatched_length")


def read_archives(tars):
    """path -> (size, 'YYYY-MM-DD HH:MM', archive) from every <name>.files.txt."""
    held, names, uploaded = {}, [], set()
    for listing in sorted(glob.glob(os.path.join(tars, "*.files.txt"))):
        name = os.path.basename(listing)[:-len(".files.txt")]
        names.append(name)
        if os.path.exists(os.path.join(tars, name + ".uploaded")):
            uploaded.add(name)
        with open(listing, errors="surrogateescape") as f:
            for line in f:
                if not line.startswith("-"):
                    continue  # directories, links
                fields = line.rstrip("\n").split(None, 5)
                if len(fields) < 6:
                    continue
                _, _, size, day, hhmm, path = fields
                held[path] = (int(size), day + " " + hhmm, name)
    return held, names, uploaded


def walk(base, tops):
    """rel path -> (size, 'YYYY-MM-DD HH:MM', mtime) for every regular file."""
    files = {}
    for top in tops:
        for root, dirs, names in os.walk(os.path.join(base, top)):
            rel_root = os.path.relpath(root, base)
            dirs[:] = [d for d in dirs if not excluded(rel_root + "/" + d)]
            for n in names:
                rel = rel_root + "/" + n
                st = os.lstat(os.path.join(root, n))
                if not stat.S_ISREG(st.st_mode):
                    continue  # links, fifos
                files[rel] = (st.st_size, time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)), st.st_mtime)
    return files


def gb(n):
    return "%.1f" % (n / 1e9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tars", default=".", help="directory of yesterday's archives (default: .)")
    ap.add_argument("--base", default=None, help="model-merging project dir (default: parent of --tars)")
    ap.add_argument("--home", default=os.path.expanduser("~/model-merging"), help="checkout holding evaluate/logs and results")
    args = ap.parse_args()
    tars = os.path.realpath(args.tars)
    base = os.path.realpath(args.base or os.path.join(tars, ".."))
    ref = os.path.join(tars, "plan.tsv")

    held, names, uploaded = read_archives(tars)
    if not held:
        sys.exit("no *.files.txt in %s" % tars)
    print("== Archives in %s: %d listed, %d uploaded and verified, %d files" % (tars, len(names), len(uploaded), len(held)))
    for n in names:
        if n not in uploaded:
            print("   NOT UPLOADED: %s" % n)

    print("== Scanning %s/{models,datasets} ..." % base, flush=True)
    disk = walk(base, ["models", "datasets"])

    # Files under mismatched_length/ identical to an archived file at the unmoved path.
    moved_from = {}
    for rel, (size, mt, _) in disk.items():
        if "/mismatched_length/" in rel:
            old = unmoved(rel)
            h = held.get(old)
            if h and h[0] == size and h[1] == mt:
                moved_from[rel] = old

    units = defaultdict(lambda: defaultdict(int))  # unit -> counter
    for rel, (size, mt, mtime) in disk.items():
        u = units[unit(rel)]
        h = held.get(rel)
        if rel in moved_from:
            kind = "moved"
        elif h is None:
            kind = "new"
        elif h[0] == size and h[1] == mt:
            kind = "same"
        else:
            kind = "changed"
        u[kind] += 1
        if kind in ("new", "changed"):
            u["copy_bytes"] += size
        u["newest"] = max(u["newest"], mtime)
    moved_targets = set(moved_from.values())
    gone = [p for p in held if p not in disk and p not in moved_targets and not excluded(p)]
    for p in gone:
        units[unit(p)]["gone"] += 1

    def expected_new(u):
        return any(re.search(p, u) for p in EXPECT_NEW)

    print("\n== Units with anything to copy, moved or gone (GB = new + changed)")
    print("%-75s %5s %5s %5s %5s %5s %7s  %-16s %s" % ("unit", "new", "chg", "same", "moved", "gone", "GB", "newest file", "check"))
    flags = []
    total = 0
    for name in sorted(units):
        c = units[name]
        to_copy = c["new"] + c["changed"]
        exp = expected_new(name)
        if re.search(r"\.failed_\d+$", name):
            note = "FAILED BUILD, kept aside"
        elif "mismatched_length" in name:
            note = "IN mismatched_length, NOT ARCHIVED" if to_copy else "ok"
        elif exp and not to_copy and not c["gone"]:
            note = "EXPECTED NEW, IS UNCHANGED"
        elif exp and c["same"]:
            note = "EXPECTED NEW, PARTLY UNCHANGED"
        elif not exp and to_copy:
            note = "UNEXPECTED CHANGE"
        elif c["gone"] and not to_copy:
            note = "GONE (deleted or moved elsewhere)"
        else:
            note = "ok"
        if note != "ok":
            flags.append((name, note))
        if not (to_copy or c["moved"] or c["gone"]) and note == "ok":
            continue  # unchanged and expected so: not listed
        total += c["copy_bytes"]
        newest = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["newest"])) if c["newest"] else "-"
        print("%-75s %5d %5d %5d %5d %5d %7s  %-16s %s" % (name[:75], c["new"], c["changed"], c["same"],
                                                       c["moved"], c["gone"], gb(c["copy_bytes"]), newest, note))
    # Archived units moved to mismatched_length/ with nothing at the old path now.
    for old_unit in sorted({unit(p) for p in moved_targets} - set(units)):
        flags.append((old_unit, "MOVED, NOT REBUILT"))
    print("\nTotal to copy (models + datasets): %s GB uncompressed" % gb(total))

    print("\n== Checks")
    print("moved to mismatched_length/ and identical to archived: %d files" % len(moved_from))
    print("archived files gone from disk, not explained by the move: %d" % len(gone))
    for p in gone[:10]:
        print("   %s  (%s)" % (p, held[p][2]))
    if len(gone) > 10:
        print("   ... and %d more (units marked gone above)" % (len(gone) - 10))
    if flags:
        print("%d units differ from the expectation:" % len(flags))
        for name, note in flags:
            print("   %-34s %s" % (note, name))
    else:
        print("every unit matches the expectation")

    # Evals: none were archived, so everything there newer than plan.tsv is new work.
    if os.path.exists(ref):
        cut = os.path.getmtime(ref)
        print("\n== Eval files newer than plan.tsv (%s), not in any archive" % time.strftime("%Y-%m-%d %H:%M", time.localtime(cut)))
        for top in ("evaluate/logs", "results"):
            per = defaultdict(lambda: [0, 0])
            for root, _, fnames in os.walk(os.path.join(args.home, top)):
                for n in fnames:
                    p = os.path.join(root, n)
                    if os.path.isfile(p) and not os.path.islink(p) and os.path.getmtime(p) > cut:
                        sub = os.path.relpath(root, os.path.join(args.home, top)).split("/")[0]
                        per[sub][0] += 1
                        per[sub][1] += os.path.getsize(p)
            print("%s/%s: %d files, %s GB" % (args.home, top, sum(v[0] for v in per.values()), gb(sum(v[1] for v in per.values()))))
            for sub in sorted(per):
                print("   %-40s %6d files %7s GB" % (sub, per[sub][0], gb(per[sub][1])))


if __name__ == "__main__":
    main()
