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
  - the XSum models and datasets of seeds 1-5 were moved to mismatched_length/,
    then regenerated and retrained;
  - task merging (dense merges and MoEs) was redone for seeds 1-3;
  - nothing else changed.
Output is a summary by area; --detail lists every unit. Last, the eval logs
and results in the home checkout are matched against the archives the same way.
"""
import argparse
import glob
import os
import re
import stat
import sys
import time
from collections import defaultdict

EXPECT_NEW = [r"^models/xsum/seed_[1-5]/", r"^models/task_merge/seed_[123]/",
              r"^datasets/xsum/(?!mismatched_length)"]  # regenerated XSum data
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


def area(u):
    """Unit -> the directory it is summarised under: everything in mismatched_length/
    together, else the unit's last seed_N / split_N directory."""
    parts = u.split("/")
    if "mismatched_length" in parts:
        return "/".join(parts[:parts.index("mismatched_length") + 1])
    hits = [i for i, p in enumerate(parts[:-1]) if GROUP_DIR.fullmatch(p)]
    return "/".join(parts[:hits[-1] + 1]) if hits else u


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
    ap.add_argument("--detail", action="store_true", help="list every unit, not just the summary by area")
    ap.add_argument("--show", type=int, default=8, help="units listed per check without --detail (default 8)")
    args = ap.parse_args()
    tars = os.path.realpath(args.tars)
    base = os.path.realpath(args.base or os.path.join(tars, ".."))
    ref = os.path.join(tars, "plan.tsv")

    held_all, names, uploaded = read_archives(tars)
    if not held_all:
        sys.exit("no *.files.txt in %s" % tars)
    held = {p: h for p, h in held_all.items() if p.startswith(("models/", "datasets/"))}
    print("== Archives in %s: %d listed, %d uploaded and verified, %d files" % (tars, len(names), len(uploaded), len(held_all)))
    for n in names:
        if n not in uploaded:
            side = [ext for ext in (".tar.zst", ".tar.zst.part", ".md5", ".verify.txt", ".azcopy.txt")
                    if os.path.exists(os.path.join(tars, n + ext))]
            first = next((p for p, h in held_all.items() if h[2] == n), "?")
            print("   NOT UPLOADED: %s  (files here: %s; e.g. holds %s)" % (n, " ".join(side) or "none", first))

    print("== Scanning %s/{models,datasets} ..." % base, flush=True)
    disk = walk(base, ["models", "datasets"])

    # Files under mismatched_length/ identical to an archived file at the unmoved path.
    moved_from = {}
    for rel, (size, mt, _) in disk.items():
        if "/mismatched_length/" in rel:
            h = held.get(unmoved(rel))
            if h and h[0] == size and h[1] == mt:
                moved_from[rel] = unmoved(rel)

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

    notes = {}
    for name, c in units.items():
        to_copy = c["new"] + c["changed"]
        exp = any(re.search(p, name) for p in EXPECT_NEW)
        if re.search(r"\.failed_\d+$", name):
            notes[name] = "FAILED BUILD, kept aside"
        elif "mismatched_length" in name:
            notes[name] = "IN mismatched_length, NOT ARCHIVED" if to_copy else "ok"
        elif exp and not to_copy and not c["gone"]:
            notes[name] = "EXPECTED NEW, IS UNCHANGED"
        elif exp and c["same"]:
            notes[name] = "EXPECTED NEW, PARTLY UNCHANGED"
        elif not exp and to_copy:
            notes[name] = "UNEXPECTED CHANGE"
        elif c["gone"] and not to_copy:
            notes[name] = "GONE (deleted or moved elsewhere)"
        else:
            notes[name] = "ok"
    # Archived units moved to mismatched_length/ with nothing at the old path now.
    for old_unit in {unit(p) for p in moved_targets} - set(units):
        units[old_unit]  # an empty row, so it is listed
        notes[old_unit] = "MOVED, NOT REBUILT"

    cols = ("new", "changed", "same", "moved", "gone")
    head = "%5s %5s %5s %5s %5s %8s  %-16s" % ("new", "chg", "same", "moved", "gone", "GB", "newest file")

    def row(c):
        newest = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["newest"])) if c["newest"] else "-"
        return "%5d %5d %5d %5d %5d %8s  %-16s" % (tuple(c[k] for k in cols) + (gb(c["copy_bytes"]), newest))

    # Summary: units grouped into areas (models/task_merge/seed_1, models/xsum/seed_1/split_2,
    # datasets/xsum/mismatched_length, ...); areas with nothing new, moved or flagged are left out.
    areas = defaultdict(lambda: defaultdict(int))
    area_notes = defaultdict(lambda: defaultdict(int))
    for name, c in units.items():
        a = area(name)
        for k in cols + ("copy_bytes",):
            areas[a][k] += c[k]
        areas[a]["newest"] = max(areas[a]["newest"], c["newest"])
        areas[a]["units"] += 1
        area_notes[a][notes[name]] += 1
    print("\n== Summary by area (GB = new + changed, uncompressed; checks = units per check)")
    print("%-50s %5s %s  %s" % ("area", "units", head, "checks"))
    for a in sorted(areas):
        c = areas[a]
        if not (c["new"] or c["changed"] or c["moved"] or c["gone"]) and set(area_notes[a]) == {"ok"}:
            continue
        checks = ", ".join("%d %s" % (n, k) for k, n in sorted(area_notes[a].items()))
        print("%-50s %5d %s  %s" % (a[:50], c["units"], row(c), checks))
    print("\nTotal to copy (models + datasets): %s GB uncompressed" % gb(sum(c["copy_bytes"] for c in units.values())))

    if args.detail:
        print("\n== Every unit with anything new, changed, moved or gone, or flagged")
        print("%-80s %s  %s" % ("unit", head, "check"))
        for name in sorted(units):
            c = units[name]
            if c["new"] or c["changed"] or c["moved"] or c["gone"] or notes[name] != "ok":
                print("%-80s %s  %s" % (name[:80], row(c), notes[name]))

    print("\n== Checks")
    print("moved to mismatched_length/ and identical to archived: %d files" % len(moved_from))
    print("archived files gone from disk, not explained by the move: %d" % len(gone))
    for p in gone[:10]:
        print("   %s  (%s)" % (p, held[p][2]))
    if len(gone) > 10:
        print("   ... and %d more" % (len(gone) - 10))
    flagged = sorted((n, u) for u, n in notes.items() if n != "ok")
    by_note = defaultdict(list)
    for n, u in flagged:
        by_note[n].append(u)
    if not by_note:
        print("every unit matches the expectation")
    for n in sorted(by_note):
        us = by_note[n]
        print("%s: %d units" % (n, len(us)))
        shown = us if args.detail or len(us) <= args.show else us[:args.show]
        for u in shown:
            print("   %s" % u)
        if len(shown) < len(us):
            print("   ... and %d more (--detail lists all)" % (len(us) - len(shown)))

    # Evals: in an archive (uploaded or not), or newer than nothing archived.
    print("\n== Eval logs and results in %s" % args.home)
    for top in ("evaluate/logs", "results"):
        per = defaultdict(lambda: defaultdict(int))
        for root, _, fnames in os.walk(os.path.join(args.home, top)):
            sub = os.path.relpath(root, os.path.join(args.home, top)).split("/")[0]
            for n in fnames:
                p = os.path.join(root, n)
                st = os.lstat(p)
                if not stat.S_ISREG(st.st_mode):
                    continue
                rel = os.path.relpath(p, args.home)  # evaluate/logs/...
                mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
                h = next((held_all[k] for k in (rel, rel.split("/", 1)[1], "model-merging/" + rel) if k in held_all), None)
                if h is None:
                    kind = "new"
                elif h[0] == st.st_size and h[1] == mt:
                    kind = "same_up" if h[2] in uploaded else "same_notup"
                else:
                    kind = "changed"
                per[sub][kind] += 1
                if kind in ("new", "changed", "same_notup"):
                    per[sub]["copy_bytes"] += st.st_size
        tot = defaultdict(int)
        for v in per.values():
            for k, n in v.items():
                tot[k] += n
        print("%s: %d not in any archive, %d changed since archived, %d archived but not uploaded, %d on Azure; %s GB to copy"
              % (top, tot["new"], tot["changed"], tot["same_notup"], tot["same_up"], gb(tot["copy_bytes"])))
        for sub in sorted(per):
            v = per[sub]
            if v["new"] or v["changed"] or v["same_notup"]:
                print("   %-45s new %5d  chg %5d  not uploaded %5d  on Azure %5d  %7s GB"
                      % (sub[:45], v["new"], v["changed"], v["same_notup"], v["same_up"], gb(v["copy_bytes"])))

if __name__ == "__main__":
    main()
