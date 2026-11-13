#!/usr/bin/env python3
"""Checks an upload_direct.sh transfer: every local file of every plan.tsv entry
is on Azure under model-merging/files_<TAG>/ with the same size and Content-MD5.

    export SAS; nohup setsid python3 verify_direct.py >> verify.log 2>&1 < /dev/null &

The MD5s are computed here, from the files on Lustre, independently of azcopy
(which set each blob's Content-MD5 from the bytes it read, --put-md5). Restartable:
each MD5 is appended to md5_done.tsv as soon as it is computed, and a rerun hashes
only files not there (or changed since: size or mtime differ). Writes manifest.tsv
(path, size, md5 base64) and verify_report.txt; exit 0 only if all match.
--no-md5 checks sizes only (a minute or two). The SAS is never printed.
"""
import argparse
import base64
import hashlib
import os
import re
import stat
import subprocess
import sys
import time
from multiprocessing import Pool

CONTAINER = ("https://tiremodelmergi5417488796.blob.core.windows.net/"
             "azureml-blobstore-064c9079-e9d7-4131-93b9-84f72e0c0e68")
EXCLUDE = re.compile(r"(^|/)(intermediate|\.lora_merge_cache[^/]*)(/|$)")  # as upload_direct.sh


def local_files(plan):
    """dest path (relative to files_<TAG>/) -> (local path, size, mtime_ns), as azcopy uploads them."""
    files = {}
    for line in open(plan):
        root, rel = line.rstrip("\n").split("\t")
        src = os.path.join(root, rel)
        if os.path.isfile(src):
            st = os.stat(src)
            files[rel] = (src, st.st_size, st.st_mtime_ns)
            continue
        parent = os.path.dirname(rel)
        for d, dirs, names in os.walk(src):
            for n in names:
                p = os.path.join(d, n)
                st = os.lstat(p)
                if not stat.S_ISREG(st.st_mode):
                    continue
                inside = os.path.relpath(p, src)        # what --exclude-regex is matched against
                if EXCLUDE.search(inside):
                    continue
                dest = os.path.join(parent, os.path.relpath(p, os.path.dirname(src)))
                files[dest] = (p, st.st_size, st.st_mtime_ns)
    return files


def md5(job):
    name, path = job
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return name, base64.b64encode(h.digest()).decode()


def remote_files(azcopy, dest_url, sas, raw_out):
    """dest path (relative to files_<TAG>/) -> (size, md5 base64 or '') from azcopy list,
    which may print each blob's path from the container root or relative to the prefix."""
    prefix = dest_url.split(".net/", 1)[1].split("/", 1)[1].rstrip("/") + "/"   # model-merging/files_<TAG>/
    cmd = [azcopy, "list", dest_url + "/?" + sas, "--properties", "ContentMD5", "--machine-readable"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
    out = re.sub(r"sig=[^&\s\"]*", "sig=<redacted>", out)
    with open(raw_out, "w") as f:
        f.write(out)
    blobs = {}
    for line in out.splitlines():
        # "<path>; ContentMD5: <b64>; Content Length: <n>" (azcopy 10.34; fields in either order)
        m = re.match(r"^(?:INFO: )?(.+?);\s*(?:ContentMD5|Content Length):", line)
        size = re.search(r"Content Length: *([0-9]+)", line)
        if not (m and size):
            continue
        md = re.search(r"ContentMD5: *([A-Za-z0-9+/=]+)", line)
        name = m.group(1).strip().lstrip("/")
        if name.startswith(prefix):
            name = name[len(prefix):]
        blobs[name] = (int(size.group(1)), md.group(1) if md else "")
    return blobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=os.environ.get("TAG", "2026-11-06"))
    ap.add_argument("--no-md5", action="store_true", help="sizes only (fast)")
    ap.add_argument("--jobs", type=int, default=16, help="files hashed at once (default 16)")
    args = ap.parse_args()
    sas = os.environ.get("SAS", "").lstrip("?")
    if not sas:
        sys.exit("SAS is not set: export SAS in this shell")
    azcopy = os.path.join(os.environ["PROJECTDIR"], "shared", "azcopy")

    files = local_files("plan.tsv")
    print("%s  %d local files, %.1f GB" % (time.strftime("%F %T"), len(files), sum(v[1] for v in files.values()) / 1e9), flush=True)
    blobs = remote_files(azcopy, CONTAINER + "/model-merging/files_" + args.tag, sas, "azcopy_list.txt")
    print("%d blobs under model-merging/files_%s/" % (len(blobs), args.tag), flush=True)
    if files and not blobs:
        sys.exit("no blobs parsed from azcopy list: see azcopy_list.txt (format, or SAS?)")
    found = sum(1 for n in files if n in blobs)
    if found < len(files) / 2:   # most missing: more likely a naming mismatch than a failed upload
        print("only %d of %d local files found by name on Azure; names compared, e.g.:" % (found, len(files)))
        for n in sorted(files)[:3]:
            print("   local: %r" % n)
        for n in sorted(blobs)[:3]:
            print("   Azure: %r" % n)

    names = sorted(files)
    sums = {}
    if not args.no_md5:
        # Resume: MD5s already computed, kept if the file is unchanged (same size and mtime).
        done_file = "md5_done.tsv"
        if os.path.exists(done_file):
            for line in open(done_file):
                f = line.rstrip("\n").split("\t")
                if len(f) == 4 and f[0] in files and files[f[0]][1:] == (int(f[1]), int(f[2])):
                    sums[f[0]] = f[3]
        todo = [n for n in names if n not in sums]
        total = sum(files[n][1] for n in names)
        resumed = sum(files[n][1] for n in sums)
        print("hashing %.1f GB of %.1f GB (%d files already in %s) with %d processes, progress every ~2 min ..."
              % ((total - resumed) / 1e9, total / 1e9, len(sums), done_file, args.jobs), flush=True)
        t0 = last = time.time()
        new = 0
        with Pool(args.jobs) as pool, open(done_file, "a") as out:
            if out.tell() and open(done_file, "rb").read()[-1:] != b"\n":
                out.write("\n")   # a line cut short by a kill: start a fresh one
            for name, h in pool.imap_unordered(md5, [(n, files[n][0]) for n in todo], chunksize=1):
                sums[name] = h
                out.write("%s\t%d\t%d\t%s\n" % (name, files[name][1], files[name][2], h))
                out.flush()
                new += files[name][1]
                now = time.time()
                if now - last >= 120 or len(sums) == len(names):
                    last = now
                    rate = new / max(now - t0, 1e-9)
                    eta = (total - resumed - new) / rate if rate else 0
                    print("  %s  %d/%d files, %.0f/%.0f GB, %.0f MB/s, ~%.0f min left"
                          % (time.strftime("%H:%M:%S"), len(sums), len(names), (resumed + new) / 1e9, total / 1e9,
                             rate / 1e6, eta / 60), flush=True)
        with open("manifest.tsv", "w") as f:
            for n in names:
                f.write("%s\t%d\t%s\n" % (n, files[n][1], sums[n]))

    problems = []
    for n in names:
        size = files[n][1]
        b = blobs.get(n)
        if b is None:
            problems.append("MISSING  %s" % n)
        elif b[0] != size:
            problems.append("SIZE     %s  local %d, Azure %d" % (n, size, b[0]))
        elif not args.no_md5 and b[1] != sums[n]:
            problems.append("MD5      %s  local %s, Azure %s" % (n, sums[n], b[1] or "none"))
    extra = sorted(set(blobs) - set(files))
    with open("verify_report.txt", "w") as f:
        for p in problems:
            f.write(p + "\n")
        for e in extra:
            f.write("EXTRA    %s  (on Azure, not in the plan)\n" % e)
    checked = "size and MD5" if not args.no_md5 else "size only"
    print("%d of %d files match (%s); %d problems; %d blobs on Azure not in the plan"
          % (len(names) - len(problems), len(names), checked, len(problems), len(extra)))
    for p in problems[:20]:
        print("  " + p)
    if len(problems) > 20:
        print("  ... see verify_report.txt")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
