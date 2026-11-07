#!/bin/bash
#SBATCH --job-name=tar_upload
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000M
#SBATCH --time=24:00:00
#SBATCH --output=tar_upload-%j.out
# Makes every archive in plan.tsv and uploads each to Azure as soon as it is
# made, in one job (one wait in the queue):
#   - make_one.sh, 3 at a time: tar | zstd -T20, checksums taken on the way;
#   - upload_one.sh, one at a time, in plan order as archives become ready:
#     azcopy copy (100 MiB blocks, 64 concurrent, --put-md5), then the blob's
#     size and Content-MD5 checked against the local archive.
# Submit from tars/, in the shell that holds the SAS token:
#   export SAS && bash make_plan.sh && sbatch make_and_upload.sbatch
# Resubmitting resumes: finished archives (<name>.tar.zst) are not remade and
# verified uploads (<name>.uploaded) are not repeated.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
AZCOPY=$PROJECTDIR/shared/azcopy
CONTAINER=https://tiremodelmergi5417488796.blob.core.windows.net/azureml-blobstore-064c9079-e9d7-4131-93b9-84f72e0c0e68
export SAS=${SAS:-}
[[ -n $SAS ]] || { echo "SAS is not set: run 'export SAS' in the shell that holds it, then resubmit" >&2; exit 1; }
[[ -f plan.tsv ]] || { echo "no plan.tsv: run make_plan.sh first" >&2; exit 1; }

# Fail fast if the token cannot list the container.
if ! "$AZCOPY" list "$CONTAINER/model-merging/?${SAS#\?}" --output-level essential > /dev/null 2>&1; then
  echo "the SAS token cannot list $CONTAINER (expired, or no list permission?)" >&2; exit 1
fi
echo "$(date -u +%T) SAS token works; $(wc -l < plan.tsv) archives in plan.tsv"

mapfile -t NAMES < <(cut -f1 plan.tsv)
printf '%s\n' "${NAMES[@]}" | xargs -P 3 -n 1 bash make_one.sh &
MAKER=$!

declare -A tries
while :; do
  progress=0
  for name in "${NAMES[@]}"; do
    [[ -f $name.uploaded || ! -f $name.tar.zst ]] && continue
    (( ${tries[$name]:-0} >= 2 )) && continue
    tries[$name]=$(( ${tries[$name]:-0} + 1 ))
    bash upload_one.sh "$name"
    rc=$?
    (( rc == 2 )) && tries[$name]=2   # uploaded but unverified: do not upload it again
    progress=1
    break   # back to the start: upload in plan order as archives become ready
  done
  if (( !progress )); then
    kill -0 "$MAKER" 2>/dev/null || break   # nothing left to make or upload
    sleep 30
  fi
done
wait "$MAKER"

# The listing of what each archive holds, and the plan, alongside the archives.
for f in CONTENTS.md plan.tsv; do
  [[ -f $f ]] && "$AZCOPY" copy "$f" "$CONTAINER/model-merging/tars/$f?${SAS#\?}" --overwrite true \
    --output-level quiet > /dev/null 2>&1
done
made=0; up=0
for name in "${NAMES[@]}"; do [[ -f $name.tar.zst ]] && made=$((made + 1)); [[ -f $name.uploaded ]] && up=$((up + 1)); done
echo "$(date -u +%T) finished: ${#NAMES[@]} in plan, $made made, $up uploaded and verified"
for name in "${NAMES[@]}"; do [[ -f $name.uploaded ]] || echo "  not done: $name"; done
