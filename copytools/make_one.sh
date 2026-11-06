#!/bin/bash
# Makes one archive from plan.tsv:  bash make_one.sh <name>   (from tars/)
#
# Paths are taken relative to the *resolved* model-merging directory, so a
# symlinked route to it (e.g. the repo's datasets -> /projects/... link) can
# never be what is archived. Refused before starting: a path that is missing,
# is a symlink, or contains one. Failed after: an archive with no regular files.
#
# Writes <name>.tar.zst (renamed from .part only on success), <name>.files.txt
# (tar's listing, with sizes), <name>.sha256 and <name>.md5 (of the archive,
# computed from the bytes as they are written, so the archive is not re-read).
set -uo pipefail
name=$1
BASE=$(realpath "$PROJECTDIR/$USER/model-merging")
paths=$(awk -F'\t' -v n="$name" '$1 == n {print $2}' plan.tsv)
fail() { rm -f "$name.tar.zst.part" "$name".*.part; echo "$(date -u +%T) FAILED $name: $*"; exit 1; }
[[ -n $paths ]] || fail "not in plan.tsv"
if [[ -f $name.tar.zst ]]; then echo "$(date -u +%T) skip   $name (exists)"; exit 0; fi
for p in $paths; do
  [[ -e $BASE/$p ]] || fail "$p does not exist"
  [[ ! -L $BASE/$p ]] || fail "$p is a symlink"
  links=$(find "$BASE/$p" -type l -print -quit)
  [[ -z $links ]] || fail "$p contains a symlink: $links"
done
echo "$(date -u +%T) start  $name"
t0=$SECONDS
# The archive's checksums are taken from the stream through named pipes
# (bash 4.4 cannot wait for process substitutions).
fifo_md5=$(mktemp -u "$name.md5fifo.XXXX"); fifo_sha=$(mktemp -u "$name.shafifo.XXXX")
mkfifo "$fifo_md5" "$fifo_sha"
md5sum < "$fifo_md5" > "$name.md5.part" & p1=$!
sha256sum < "$fifo_sha" > "$name.sha256.part" & p2=$!
# shellcheck disable=SC2086  # paths are space-separated by design
tar -C "$BASE" -cvvf - --index-file="$name.files.txt.part" \
    --exclude='.lora_merge_cache*' --exclude='intermediate' $paths \
  | zstd -q -T20 \
  | tee "$fifo_md5" "$fifo_sha" > "$name.tar.zst.part"
status=$?
wait $p1; s1=$?; wait $p2; s2=$?
rm -f "$fifo_md5" "$fifo_sha"
(( status == 0 && s1 == 0 && s2 == 0 )) || fail "exit $status (md5 $s1, sha256 $s2)"
nfiles=$(grep -c '^-' "$name.files.txt.part" || true)
bytes=$(awk '$1 ~ /^-/ {s += $3} END {print s + 0}' "$name.files.txt.part")
(( nfiles > 0 && bytes > 0 )) || fail "no regular files archived"
sed -i "s|  -\$|  $name.tar.zst|" "$name.md5.part" "$name.sha256.part"
mv "$name.files.txt.part" "$name.files.txt"
mv "$name.md5.part" "$name.md5"
mv "$name.sha256.part" "$name.sha256"
mv "$name.tar.zst.part" "$name.tar.zst"
awk -v d="$(date -u +%T)" -v n="$name" -v f="$nfiles" -v b="$bytes" -v out="$(stat -c %s "$name.tar.zst")" \
    -v t=$((SECONDS - t0)) 'BEGIN {printf "%s done   %s: %d files, %.1f GB -> %.1f GB, %d s\n", d, n, f, b/1e9, out/1e9, t}'
