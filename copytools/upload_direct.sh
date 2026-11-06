#!/bin/bash
# Uploads every entry of plan.tsv (make_plan_direct.sh) to Azure as plain files,
# one azcopy job per entry, in order, on a login node (no Slurm needed):
#   export SAS                      # the container SAS token, in this shell
#   nohup setsid bash upload_direct.sh >> upload.log 2>&1 < /dev/null &
#   tail -f upload.log              # from any login node: the log is on Lustre
# Destination: <container>/model-merging/files_<TAG>/<path>, TAG default 2026-11-06.
# --overwrite false: nothing already on Azure is ever replaced, so the 5 Nov tars/
# are safe, and rerunning resumes (files already uploaded are skipped). An entry
# is marked done (done/<entry>) when azcopy finishes it with no failed file.
# Then check everything with verify_direct.py. The SAS is never printed.
set -uo pipefail
TAG=${TAG:-2026-11-06}
AZCOPY=$PROJECTDIR/shared/azcopy
CONTAINER=https://tiremodelmergi5417488796.blob.core.windows.net/azureml-blobstore-064c9079-e9d7-4131-93b9-84f72e0c0e68
DEST=$CONTAINER/model-merging/files_$TAG
SAS=${SAS:-}; SAS=${SAS#\?}
EXCLUDE='(^|/)(intermediate|\.lora_merge_cache[^/]*)(/|$)'
redact() { sed -E 's/\?[^ "]*sig=[^ "]*/?<SAS>/g; s/sig=[^& "]*/sig=<redacted>/g'; }
log() { echo "$(date -u +%FT%TZ) $*"; }

[[ -n $SAS ]] || { log "SAS is not set: export SAS in this shell, then rerun"; exit 1; }
[[ -f plan.tsv ]] || { log "no plan.tsv: run make_plan_direct.sh first"; exit 1; }
mkdir -p done azcopy_logs azcopy_plans
export AZCOPY_LOG_LOCATION=$PWD/azcopy_logs AZCOPY_JOB_PLAN_LOCATION=$PWD/azcopy_plans
export AZCOPY_CONCURRENCY_VALUE=${AZCOPY_CONCURRENCY_VALUE:-64} AZCOPY_BUFFER_GB=${AZCOPY_BUFFER_GB:-8}

# One copy at a time: a second run would race the first.
exec 9> upload.lock
flock -n 9 || { log "another upload_direct.sh is running here (upload.lock)"; exit 1; }

expiry=$(grep -oE '(^|&)se=[^&]*' <<<"$SAS" | sed 's/.*se=//; s/%3A/:/g')
log "start: $(wc -l < plan.tsv) entries -> model-merging/files_$TAG/; SAS expires ${expiry:-?} (UTC)"
# Other users can read the command line (and so the SAS) of our processes unless /proc is mounted hidepid.
findmnt -no OPTIONS /proc 2>/dev/null | grep -q hidepid || log "note: /proc has no hidepid; the SAS is visible in azcopy's command line to other users of this node"
if ! "$AZCOPY" list "$CONTAINER/model-merging/tars/?$SAS" --output-level essential > /dev/null 2>&1; then
  log "the SAS token cannot list the container (expired, or no list permission?)"; exit 1
fi

n=0; ok=0; failed=()
while IFS=$'\t' read -r root rel; do
  n=$((n + 1))
  key=${rel//\//%}
  if [[ -f done/$key ]]; then ok=$((ok + 1)); continue; fi
  src=$root/$rel
  if [[ -d $src ]]; then dst=$DEST/$(dirname "$rel"); rec=(--recursive); else dst=$DEST/$rel; rec=(); fi
  log "[$n] $rel ($(du -sh "$src" 2>/dev/null | cut -f1))"
  t0=$SECONDS
  "$AZCOPY" copy "$src" "$dst?$SAS" ${rec[@]+"${rec[@]}"} --overwrite false --put-md5 --block-size-mb 100 \
      --exclude-regex "$EXCLUDE" --log-level WARNING --output-level essential < /dev/null 2>&1 \
    | redact > "azcopy_logs/$key.txt"
  rc=${PIPESTATUS[0]}
  status=$(sed -nE 's/.*Final Job Status: *([A-Za-z]+).*/\1/p' "azcopy_logs/$key.txt" | tail -1)
  nfail=$(sed -nE 's/.*Number of (File )?Transfers Failed: *([0-9]+).*/\2/p' "azcopy_logs/$key.txt" | tail -1)
  if (( rc == 0 )) && [[ $status == Completed || $status == CompletedWithSkipped ]] && [[ ${nfail:-0} == 0 ]]; then
    grep -E 'Transfers|Bytes|Elapsed|Final Job Status' "azcopy_logs/$key.txt" > "done/$key"
    ok=$((ok + 1))
    log "    ok, $((SECONDS - t0)) s ($status)"
  else
    failed+=("$rel")
    log "    FAILED (exit $rc, ${status:-no status}, ${nfail:-?} failed files): see azcopy_logs/$key.txt"
    tail -5 "azcopy_logs/$key.txt" | sed 's/^/      /'
  fi
done < plan.tsv
log "finished: $ok of $n entries done"
for r in ${failed[@]+"${failed[@]}"}; do log "  not done: $r"; done
(( ${#failed[@]} == 0 )) && log "next: python3 verify_direct.py"
