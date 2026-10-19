#!/bin/bash
# Download Qwen/Qwen3-32B (the DeepEval judge) at a pinned revision with curl:
# the HF client has hung on locks on this machine, and Xet downloads are
# memory-hungry. Resumable: complete files are skipped, partial ones resumed.
set -euo pipefail
D=$PROJECTDIR/$USER/mm/models/qwen3-32b
mkdir -p "$D"; cd "$D"
curl -sfL "https://huggingface.co/api/models/Qwen/Qwen3-32B?blobs=true" > .api.json
python3 - <<PY > .files
import json
d = json.load(open(".api.json"))
open("REVISION", "w").write(d["sha"] + "\n")
for s in d["siblings"]:
    print(s["rfilename"], s.get("size", -1))
PY
REV=$(cat REVISION)
while read -r f size; do
  [[ -f $f && $(stat -c %s "$f") == "$size" ]] && continue
  mkdir -p "$(dirname "$f")"
  curl -sfL --retry 5 -C - -o "$f" "https://huggingface.co/Qwen/Qwen3-32B/resolve/$REV/$f"
  [[ $(stat -c %s "$f") == "$size" || $size == -1 ]] || { echo "size mismatch: $f" >&2; exit 1; }
done < .files
echo "revision $REV: $(wc -l < .files) files, $(du -sh . | cut -f1)"
