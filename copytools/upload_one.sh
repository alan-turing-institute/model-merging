#!/bin/bash
# Uploads one finished archive and its side files to Azure, then verifies it:
#   bash upload_one.sh <name>        (from tars/, with $SAS exported)
#
# Destination: <container>/model-merging/tars/<name>.tar.zst (+ .sha256, .md5,
# .files.txt). Verified when azcopy reports success and the blob's size and
# Content-MD5 equal the local file's size and the MD5 taken while the archive
# was written (make_one.sh). Then <name>.uploaded records it, and the archive
# is skipped from then on. Exit 0 uploaded and verified; 1 upload failed;
# 2 uploaded but not verified (azcopy list output kept in <name>.verify.txt).
# The SAS is never printed: azcopy's output is passed through redact().
set -uo pipefail
name=$1
AZCOPY=$PROJECTDIR/shared/azcopy
CONTAINER=https://tiremodelmergi5417488796.blob.core.windows.net/azureml-blobstore-064c9079-e9d7-4131-93b9-84f72e0c0e68
DEST=$CONTAINER/model-merging/tars
SAS=${SAS#\?}
redact() { sed -E 's/\?[^ "]*sig=[^ "]*/?<SAS>/g; s/sig=[^& "]*/sig=<redacted>/g'; }
[[ -n $SAS ]] || { echo "FAILED $name: SAS is not set (export SAS before sbatch)"; exit 1; }
[[ -f $name.uploaded ]] && { echo "$(date -u +%T) skip   $name (uploaded)"; exit 0; }

export AZCOPY_CONCURRENCY_VALUE=${AZCOPY_CONCURRENCY_VALUE:-64}
export AZCOPY_BUFFER_GB=${AZCOPY_BUFFER_GB:-16}
export AZCOPY_LOG_LOCATION=$PWD/azcopy_logs AZCOPY_JOB_PLAN_LOCATION=$PWD/azcopy_plans
mkdir -p "$AZCOPY_LOG_LOCATION" "$AZCOPY_JOB_PLAN_LOCATION"

size=$(stat -c %s "$name.tar.zst")
echo "$(date -u +%T) upload $name ($(awk -v s="$size" 'BEGIN {printf "%.1f GB", s/1e9}'))"
t0=$SECONDS
for f in "$name.tar.zst" "$name.sha256" "$name.md5" "$name.files.txt"; do
  if ! "$AZCOPY" copy "$f" "$DEST/$f?$SAS" --block-size-mb 100 --put-md5 --overwrite true \
         --log-level WARNING --output-level essential 2>&1 | redact > "$name.azcopy.txt"; then
    echo "$(date -u +%T) FAILED upload $f:"; tail -5 "$name.azcopy.txt"; exit 1
  fi
done
secs=$((SECONDS - t0))
rate=$(awk -v s="$size" -v t="$secs" 'BEGIN {printf "%.0f MB/s", s/1e6/(t > 0 ? t : 1)}')

# Verify: the blob's length and Content-MD5 against the local file.
"$AZCOPY" list "$DEST/?$SAS" --properties "ContentMD5" --machine-readable 2>&1 | redact > "$name.verify.txt"
line=$(grep -E "(^|[ /])$name\.tar\.zst;" "$name.verify.txt" | head -1)
remote_size=$(sed -nE 's/.*Content Length: *([0-9]+).*/\1/p' <<<"$line")
remote_md5=$(sed -nE 's/.*ContentMD5: *([A-Za-z0-9+\/=]+).*/\1/p' <<<"$line")
hex=$(cut -d' ' -f1 "$name.md5")
local_md5=$(printf '%b' "$(sed 's/../\\x&/g' <<<"$hex")" | base64)
if [[ $remote_size == "$size" && $remote_md5 == "$local_md5" ]]; then
  printf '%s\tsize %s\tmd5 %s\t%s s\t%s\n' "$(date -u +%FT%TZ)" "$size" "$local_md5" "$secs" "$rate" > "$name.uploaded"
  rm -f "$name.verify.txt" "$name.azcopy.txt"
  echo "$(date -u +%T) ok     $name: $secs s, $rate, size and MD5 match"
  exit 0
fi
echo "$(date -u +%T) UNVERIFIED $name: local size $size md5 $local_md5; remote '${remote_size:-?}' '${remote_md5:-?}' (see $name.verify.txt)"
exit 2
