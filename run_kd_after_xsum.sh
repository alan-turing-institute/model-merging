#!/usr/bin/env bash
# Wait for the XSum pipeline to finish, then run the KD arm on XSum.
#
# WAITS FOR THE PROCESS TO EXIT, not merely for the artifacts to be registered.
# run_xsum_pipeline.sh deletes models/ as its last act, and the KD arm keeps the
# merge it distils onto in that same directory - so starting on "registered"
# races the cleanup and can have the student's init deleted mid-run.
#
# Versions come from what the XSum run actually recorded, not from
# run_kd_pipeline.sh's defaults, which predate several re-runs.
#
# Usage, from the repo root, detached:
#   screen -dmS kdx bash -lc 'exec > ~/kdx-run.log 2>&1; ./run_kd_after_xsum.sh'
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$PWD

VERSIONS_FILE=$REPO_ROOT/.pipeline_versions_xsum
POLL_SECONDS=${POLL_SECONDS:-60}
TIMEOUT_HOURS=${TIMEOUT_HOURS:-8}

# Which distillation arm to run once the XSum pipeline is done.
KD_MODE=${KD_MODE:-offline}

# How to tell that the XSum run has finished.
#
#   process   (default) poll for run_xsum_pipeline.sh on THIS machine. Right
#             when both run on the same box.
#   registry  poll the Azure ML registry for a merge version newer than
#             MIN_MERGE_VERSION. Use this on a SECOND machine: it cannot see
#             the first one's processes or its versions file, and the registry
#             is the only state the two share.
#
# MIN_MERGE_VERSION must be the highest version that exists BEFORE the run you
# are waiting on. Getting it wrong means starting immediately against whatever
# is already registered - which right now is a set of models that never learned
# to terminate.
WAIT_MODE=${WAIT_MODE:-process}
MIN_MERGE_VERSION=${MIN_MERGE_VERSION:-0}
RG=${AZUREML_RG:-tire-1}
WS=${AZUREML_WS:-tire-2}

log() { echo "=== $* ===" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

NEEDED="gemma3-xsum-1-of-2-lora gemma3-xsum-2-of-2-lora gemma3-xsum-merged-linear gemma3-xsum-full-lora"

deadline=$(( $(date +%s) + TIMEOUT_HOURS * 3600 ))

registry_version() {
  az ml model list --name "$1" --resource-group "$RG" --workspace-name "$WS" \
    --query "[].version" -o tsv 2>/dev/null | sort -n | tail -1
}

if [ "$WAIT_MODE" = "registry" ]; then
  log "Waiting for gemma3-xsum-merged-linear > v$MIN_MERGE_VERSION in $RG/$WS"
  while :; do
    current=$(registry_version gemma3-xsum-merged-linear)
    if [ -n "$current" ] && [ "$current" -gt "$MIN_MERGE_VERSION" ]; then
      log "Merge v$current is registered"
      break
    fi
    [ "$(date +%s)" -lt "$deadline" ] \
      || die "Timed out after ${TIMEOUT_HOURS}h waiting for a merge newer than v$MIN_MERGE_VERSION"
    sleep "$POLL_SECONDS"
  done

  # Take the newest of each from the registry - there is no local versions file
  # on a second machine.
  HALF1_VERSION=$(registry_version gemma3-xsum-1-of-2-lora)
  HALF2_VERSION=$(registry_version gemma3-xsum-2-of-2-lora)
  MERGE_VERSION=$(registry_version gemma3-xsum-merged-linear)
  FULL_VERSION=$(registry_version gemma3-xsum-full-lora)

  log "Starting the XSum KD arm (KD_MODE=$KD_MODE) with:"
  log "  halves v$HALF1_VERSION / v$HALF2_VERSION, merge v$MERGE_VERSION, full v$FULL_VERSION"

  KD_MODE="$KD_MODE" TASK=xsum \
  HALF1_VERSION="$HALF1_VERSION" HALF2_VERSION="$HALF2_VERSION" \
  MERGE_VERSION="$MERGE_VERSION" FULL_VERSION="$FULL_VERSION" \
    ./run_kd_pipeline.sh
  exit $?
fi

if pgrep -f "run_xsum_pipeline.sh" >/dev/null 2>&1; then
  log "Waiting for run_xsum_pipeline.sh to exit (polling every ${POLL_SECONDS}s)"
  while pgrep -f "run_xsum_pipeline.sh" >/dev/null 2>&1; do
    [ "$(date +%s)" -lt "$deadline" ] \
      || die "Timed out after ${TIMEOUT_HOURS}h waiting for the XSum pipeline to exit"
    sleep "$POLL_SECONDS"
  done
  log "XSum pipeline has exited"
else
  log "No XSum pipeline running"
fi

# Exiting is not the same as succeeding - it may have died on a hung trainer.
# Check what it actually registered before committing hours to a KD run.
missing=""
for name in $NEEDED; do
  grep -q "^$name=" "$VERSIONS_FILE" 2>/dev/null || missing="$missing $name"
done
[ -z "$missing" ] || die "The XSum run did not register:$missing
  It exited without completing. Fix and re-run it before starting the KD arm;
  distilling onto a merge that was never produced is not a thing to retry blind."

version_of() { grep "^$1=" "$VERSIONS_FILE" | tail -1 | cut -d= -f2; }

HALF1_VERSION=$(version_of gemma3-xsum-1-of-2-lora)
HALF2_VERSION=$(version_of gemma3-xsum-2-of-2-lora)
MERGE_VERSION=$(version_of gemma3-xsum-merged-linear)
FULL_VERSION=$(version_of gemma3-xsum-full-lora)

log "Starting the XSum KD arm with:"
log "  halves v$HALF1_VERSION / v$HALF2_VERSION, merge v$MERGE_VERSION, full v$FULL_VERSION"

KD_MODE="$KD_MODE" TASK=xsum \
HALF1_VERSION="$HALF1_VERSION" \
HALF2_VERSION="$HALF2_VERSION" \
MERGE_VERSION="$MERGE_VERSION" \
FULL_VERSION="$FULL_VERSION" \
  ./run_kd_pipeline.sh
