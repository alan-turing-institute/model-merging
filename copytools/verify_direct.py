#!/usr/bin/env python3
"""Checks an upload_direct.sh transfer: every local file of every plan.tsv entry
is on Azure under model-merging/files_<TAG>/ with the same size and Content-MD5.

    export SAS; python3 verify_direct.py [--no-md5] [--jobs 16]    (from the transfer dir)

The MD5s are computed here, from the files on Lustre, independently of azcopy
(which set each blob's Content-MD5 from the bytes it read, --put-md5). Writes
manifest.tsv (path, size, md5 base64) and verify_report.txt; exit 0 only if all match.
The SAS is never printed.
"""
import argparse
import base64
import hashlib
import os
import re
import stat
import subprocess
import sys
from multiprocessing import Pool

CONTAINER = ("https://tiremodelmergi5417488796.blob.core.windows.net/"
             "azureml-blobstore-064c9079-e9d7-4131-93b9-84f72e0c0e68")
EXCLUDE = re.compile(r"(^|/)(intermediate|\.lora_merge_cache[^/]*)(/|$)")  # as upload_direct.sh


def local_files(plan):
    """dest path (relative to files_<TAG>/) -> (local path, size), as azcopy uploads them."""
    files = {}
    for line in open(plan):
        root, rel = line.rstrip("\n").split("\t")
        src = os.path.join(root, rel)
        if os.path.isfile(src):
            files[rel] = (src, os.path.getsize(src))
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
                files[dest] = (p, st.st_size)
    return files


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def remote_files(azcopy, dest_url, sas, raw_out):
    """dest path -> (size, md5 base64 or '') from azcopy list."""
    cmd = [azcopy, "list", dest_url + "/?" + sas, "--properties", "ContentMD5", "--machine-readable"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
    out = re.sub(r"sig=[^&\s\"]*", "sig=<redacted>", out)
    with open(raw_out, "w") as f:
        f.write(out)
    blobs = {}
    for line in out.splitlines():
        m = re.match(r"^(?:INFO: )?(.+?);\s*Content Length: *([0-9]+)", line)
        if not m:
            continue
        md = re.search(r"ContentMD5: *([A-Za-z0-9+/=]+)", line)
        blobs[m.group(1)] = (int(m.group(2)), md.group(1) if md else "")
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
    print("%d local files, %.1f GB" % (len(files), sum(s for _, s in files.values()) / 1e9), flush=True)
    blobs = remote_files(azcopy, CONTAINER + "/model-merging/files_" + args.tag, sas, "azcopy_list.txt")
    print("%d blobs under model-merging/files_%s/" % (len(blobs), args.tag), flush=True)
    if files and not blobs:
        sys.exit("no blobs parsed from azcopy list: see azcopy_list.txt (format, or SAS?)")

    names = sorted(files)
    sums = {}
    if not args.no_md5:
        print("hashing with %d processes ..." % args.jobs, flush=True)
        with Pool(args.jobs) as pool:
            for name, h in zip(names, pool.imap(md5, [files[n][0] for n in names], chunksize=4)):
                sums[name] = h
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
