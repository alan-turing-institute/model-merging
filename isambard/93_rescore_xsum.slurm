#!/bin/bash
#SBATCH --job-name=rescore-xsum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=115000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/rescore-xsum-%j.out
#SBATCH --error=logs/rescore-xsum-%j.out
#SBATCH --account=brics.u6ui
#SBATCH --partition=workq
#
# Is the XSum null an artefact of ROUGE?
#
# The programme reports that no merge method and no distillation weighting beats
# the merge by more than noise on XSum. The fair objection is that ROUGE is
# lexical and saturated on a task whose whole full-data gap is 0.0072, so the
# null might belong to the metric rather than to the models. The metric bake-off
# established that ROUGE scores randomly shuffled words 1.000 and a wrong name
# 0.947, which makes the objection serious.
#
# This re-scores the SAME stored generations under six further metrics from
# three families. Nothing is regenerated - every arm already carries its
# predictions - so this is scoring only. If the arms separate under a metric
# ROUGE cannot resolve, the null is a measurement artefact; if they stay flat
# under all of them, it is about the models.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source isambard/config.sh
mkdir -p logs "$RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no nvidia-smi)"

ARMS="full half1 half2 merge
      kd-parity kd-t3 kd-t9 kd-aligned kd2-a09
      sweep-xsum-linear sweep-xsum-slerp sweep-xsum-task_arithmetic
      sweep-xsum-dare_ties sweep-xsum-ties sweep-xsum-arcee_fusion sweep-xsum-model_stock
      merge-4way shard1-of-4 kd4-parity
      ctx-merge ctx-kd-parity"

uv run --project evaluate python experiments/rescore_xsum.py \
  --results-dir "$RESULTS_DIR" \
  --arms $ARMS \
  --dataset "$DATASETS_DIR/xsum_dataset/test" \
  --output  "$RESULTS_DIR/xsum-rescored.json" \
  --limit 200
