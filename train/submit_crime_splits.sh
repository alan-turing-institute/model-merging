#!/bin/bash
# Trains one crime LoRA adapter per part of each data-parallel split given
# (default 2 4 8), all with crime_gemma.yaml: train.sbatch replaces its dataset
# and output paths, which are the only differences between the crime yamls.
#   ../datasets/crime/split_<N>/crime_dataset_<i>_of_<N>    from ../split_dataset.py
#     -> ../models/crime/split_<N>/gemma3_4b_crime_<i>_of_<N>
# Merging and evaluating them is separate: see ../merge/make_merge_configs.py,
# which prints the merge commands.
#
# Usage, from train/:
#   bash submit_crime_splits.sh [N...]            # dry run: checks the paths, prints the commands
#   bash submit_crime_splits.sh --submit [N...]   # submits, and appends each job to the job map
#
# A part that is already trained (its provenance.json exists and its health
# check passed) is skipped. An output dir without one is left from a failed or
# running job: the script stops rather than overwrite it, so move it aside
# first. The job map (job id, model, dataset) goes to ../evaluate/slurm/, next
# to the other job maps, so the evaluations can be matched to the models later.
set -euo pipefail
cd "$(dirname "$0")"
SUBMIT=false
if [[ ${1:-} == --submit ]]; then SUBMIT=true; shift; fi
SPLITS=("${@:-2 4 8}")
SPLITS=(${SPLITS[*]})
CONFIG=crime_gemma.yaml
JOB_MAP=../evaluate/slurm/crime_split_train_jobs_$(date +%F).txt
# Parallel jobs each running `uv sync` reinstall packages over one another.
export UV_NO_SYNC=1

for N in "${SPLITS[@]}"; do
  for ((i = 1; i <= N; i++)); do
    DATA=../datasets/crime/split_$N/crime_dataset_${i}_of_$N
    OUT=../models/crime/split_$N/gemma3_4b_crime_${i}_of_$N
    if [[ ! -d $DATA/train || ! -d $DATA/validation ]]; then
      echo "MISSING  $DATA (needs train/ and validation/: python ../split_dataset.py $N ../datasets/crime/crime_dataset ../datasets/crime/split_$N)" >&2
      exit 1
    fi
    if [[ -f $OUT/provenance.json ]]; then
      if python3 -c "import json, sys; sys.exit(0 if json.load(open('$OUT/provenance.json'))['health']['ok'] else 1)"; then
        echo "done     $OUT"
        continue
      fi
      echo "UNHEALTHY $OUT: its health check failed; move it aside to retrain it" >&2
      exit 1
    fi
    if [[ -e $OUT ]]; then
      echo "INCOMPLETE $OUT has no provenance.json (failed or still running): move it aside" >&2
      exit 1
    fi
    CMD=(sbatch --parsable --job-name="crime_${i}_of_$N" train.sbatch "$CONFIG" "$DATA" "$OUT")
    if $SUBMIT; then
      JOB=$("${CMD[@]}")
      echo "$JOB $OUT $DATA" | tee -a "$JOB_MAP"
    else
      echo "${CMD[*]}"
    fi
  done
done
$SUBMIT || echo "Dry run: nothing submitted. Rerun with --submit."
