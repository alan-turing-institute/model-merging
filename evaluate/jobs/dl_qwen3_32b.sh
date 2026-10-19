#!/bin/bash
# Download Qwen/Qwen3-32B (the DeepEval judge) at a pinned revision with curl:
# the HF client has hung on locks on this machine, and Xet downloads are
# memory-hungry. Resumable and safe to re-submit: a file of the right size is
# skipped, a partial one is resumed, and each weight shard is checked against
# the SHA-256 the Hub publishes for it.
#
# The first attempt died at 39 of 65 GB on a mid-transfer error, which curl's
# own --retry does not cover (and its retries would not recompute the -C -
# offset), so each file gets up to ten fresh resumed attempts here instead.
#   sbatch --job-name=dl-qwen3-32b --partition=workq --gres=gpu:1 --cpus-per-task=4 \
#     --mem=16000M --time=03:00:00 --output=dl-qwen3-%j.out jobs/dl_qwen3_32b.sh
set -euo pipefail
D=$PROJECTDIR/$USER/mm/models/qwen3-32b
mkdir -p "$D"; cd "$D"
if [[ ! -s REVISION ]]; then
  curl -sfL "https://huggingface.co/api/models/Qwen/Qwen3-32B?blobs=true" > .api.json
fi
# name, size, sha256 (or -) per file; the revision is fixed by the first run.
python3 - <<'PY' > .files
import json, os
d = json.load(open(".api.json"))
if not os.path.exists("REVISION"):
    open("REVISION", "w").write(d["sha"] + "\n")
for s in d["siblings"]:
    print(s["rfilename"], s.get("size", -1), (s.get("lfs") or {}).get("sha256", "-"))
PY
REV=$(cat REVISION)
while read -r f size sha; do
  if [[ ! ( -f $f && $(stat -c %s "$f") == "$size" ) ]]; then
    mkdir -p "$(dirname "$f")"
    for attempt in $(seq 10); do
      rc=0
      curl -sSfL -C - -o "$f" "https://huggingface.co/Qwen/Qwen3-32B/resolve/$REV/$f" || rc=$?
      [[ $rc == 0 && $(stat -c %s "$f") == "$size" ]] && break
      echo "$f: attempt $attempt stopped (curl $rc) at $(stat -c %s "$f" 2>/dev/null || echo 0) of $size bytes" >&2
      [[ $attempt == 10 ]] && { echo "$f: giving up" >&2; exit 1; }
      sleep 15
    done
  fi
  if [[ $sha != - ]]; then
    echo "$sha  $f" | sha256sum -c --quiet - || { echo "$f: checksum mismatch - delete it and re-submit" >&2; exit 1; }
  fi
  echo "ok $f"
done < .files
echo "revision $REV: $(wc -l < .files) files, $(du -sh . | cut -f1)"
