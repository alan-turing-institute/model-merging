#!/bin/bash
# Writes plan.tsv: one line per tar, "<name>\t<paths, space-separated>", in the
# order they are made. Paths are relative to the project's real model-merging
# directory ($PROJECTDIR/$USER/model-merging, resolved), so archives extract as
# models/... and datasets/... Run from tars/.
#
# What is archived (5 Nov):
# - datasets: all of it.
# - wine, clinc150: the MLP-only task-merging experts only (*_mlp, *_mlp_full).
# - crime, xsum: per seed, the data-parallel models (Gemma); the MLP experts in
#   their own tars; XSum's splits each in their own tar.
# - task_merge: per seed (merges and MoEs of the MLP experts, all three models).
# Left out: Mistral and Llama models that are not MLP experts (the single-task
# attention models), Gemma's attention models for wine and CLINC150, the runs
# from before the seed repeats (crime/, xsum/ top level), and the regenerable
# merge caches (.lora_merge_cache*, intermediate/; excluded by make_one.sh).
set -euo pipefail
OUT=$PWD/plan.tsv
cd "$(realpath "$PROJECTDIR/$USER/model-merging")"
MLP='_mlp(_full)?$'
OTHER='/(mistral_7b|llama3_8b)_'   # non-MLP Mistral/Llama models: left out
{
  echo -e "datasets\tdatasets"
  echo -e "wine_mlp\t$(ls -d models/wine/seed_*/* | grep -E "$MLP" | tr '\n' ' ')"
  echo -e "clinc150_mlp\t$(ls -d models/clinc150/seed_*/* | grep -E "$MLP" | tr '\n' ' ')"
  for s in 1 2 3 4 5; do
    echo -e "crime_seed_$s\t$(ls -d models/crime/seed_$s/* | grep -vE "$MLP" | grep -vE "$OTHER" | tr '\n' ' ')"
    echo -e "crime_seed_${s}_mlp\t$(ls -d models/crime/seed_$s/* | grep -E "$MLP" | tr '\n' ' ')"
  done
  for s in 1 2 3 4 5; do
    echo -e "task_merge_seed_$s\tmodels/task_merge/seed_$s"
  done
  for s in 1 2 3 4 5; do
    echo -e "xsum_seed_$s\t$(ls -d models/xsum/seed_$s/* | grep -vE "$MLP|/split_" | grep -vE "$OTHER" | tr '\n' ' ')"
    for n in 2 4 8; do echo -e "xsum_seed_${s}_split_$n\tmodels/xsum/seed_$s/split_$n"; done
    echo -e "xsum_seed_${s}_mlp\t$(ls -d models/xsum/seed_$s/* | grep -E "$MLP" | tr '\n' ' ')"
  done
} | sed 's/ *$//' > "$OUT"
