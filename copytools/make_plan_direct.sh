#!/bin/bash
# Writes plan.tsv for upload_direct.sh: one line per directory to upload,
# "<root>\t<path>", <path> relative to <root> and kept as the path on Azure
# (<container>/model-merging/files_<TAG>/<path>). Run from the transfer
# directory, e.g. $PROJECTDIR/$USER/model-merging/transfer_2026-11-06.
#
# What goes (6 Nov 2026), the work since the 5 Nov archives in tars/:
# - task merging, seeds 1-3: every merge and MoE, rebuilt with the retrained XSum
#   expert (not the *.failed_* builds kept aside, nor intermediate/ or merge caches);
# - XSum, seeds 1-5: the retrained models (Gemma, the MLP experts, the split
#   experts), as on 5 Nov leaving out non-MLP Mistral/Llama models;
# - the random-router task MoEs, seeds 1-5, built 5 Nov with the old XSum expert
#   and now in models/task_merge/mismatched_length/ (in no archive);
# - the eval logs and results, from the home checkout (in no archive).
# Not here: the XSum datasets (regenerated; uploaded by hand on 5 Nov), everything
# already in tars/ (incl. the rest of mismatched_length/), the non-MLP Mistral
# XSum models in mismatched_length/ (left out on 5 Nov too), the XSum split merges
# and task merging seeds 4-5 (not rebuilt yet).
set -euo pipefail
BASE=$(realpath "$PROJECTDIR/$USER/model-merging")
HOMEMM=$(realpath "$HOME/model-merging")
OUT=$PWD/plan.tsv
MLP='_mlp(_full)?$'
OTHER='/(mistral_7b|llama3_8b)_[^/]*$'   # non-MLP Mistral/Llama models: left out

cd "$BASE"
{
  for s in 1 2 3; do
    ls -d models/task_merge/seed_$s/* | grep -vE '/intermediate$|/\.lora_merge_cache|\.failed_[0-9]+$'
  done
  for s in 1 2 3 4 5; do
    ls -d models/xsum/seed_$s/* | grep -v '/split_[0-9]*$' | grep -vE "$OTHER" || true
    ls -d models/xsum/seed_$s/* | grep -E "$OTHER" | grep -E "$MLP" || true
    ls -d models/xsum/seed_$s/split_*/*
  done
  ls -d models/task_merge/mismatched_length/seed_*/*_task_moe_4_random
} | sort -u | awk -v r="$BASE" '{print r "\t" $0}' > "$OUT"
printf '%s\t%s\n' "$HOMEMM" evaluate/logs "$HOMEMM" results >> "$OUT"

# Refuse symlinks (azcopy would skip them silently) and missing paths.
bad=0
while IFS=$'\t' read -r root rel; do
  if [[ ! -e $root/$rel ]]; then echo "missing: $root/$rel" >&2; bad=1; continue; fi
  if [[ -L $root/$rel ]]; then echo "symlink: $root/$rel" >&2; bad=1; continue; fi
  l=$(find "$root/$rel" -type l -print -quit)
  [[ -z $l ]] || { echo "contains a symlink: $l" >&2; bad=1; }
done < "$OUT"
(( bad == 0 )) || { echo "fix the above, then rerun" >&2; exit 1; }

echo "$(wc -l < "$OUT") entries in $OUT; sizes (GB, uncompressed):"
while IFS=$'\t' read -r root rel; do
  case $rel in
    models/task_merge/mismatched_length/*) g="task_merge random MoEs, old XSum (seeds 1-5)" ;;
    models/task_merge/seed_*) g="task_merge ${rel#models/task_merge/}"; g=${g%%/*} ;;
    models/xsum/seed_*) g="xsum ${rel#models/xsum/}"; g=${g%%/*} ;;
    *) g=$rel ;;
  esac
  printf '%s\t%s\n' "$g" "$(du -sb --exclude=intermediate --exclude='.lora_merge_cache*' "$root/$rel" | cut -f1)"
done < "$OUT" | awk -F'\t' '{s[$1] += $2; t += $2} END {for (g in s) printf "  %-50s %8.1f\n", g, s[g]/1e9; printf "  %-50s %8.1f\n", "TOTAL", t/1e9}' | sort
