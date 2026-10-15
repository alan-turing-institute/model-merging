#!/bin/bash
# Data-parallel merges of gold-target KD students (CE 0.75, 4B teacher under a
# house-style prompt) on one summarisation dataset, from self-distill/:
#   ./run_gold_kd_merges.sh billsum_split|cnndm|samsum
# Chain: prompts (here) -> teacher scores the references -> axolotl preprocess ->
# the full student, and two halves of the prepared rows -> a student per half at
# seeds 42, 2 and 3 -> linear/TIES/DARE-TIES/task-arithmetic merge of each pair ->
# Inspect evals of the untrained 4B, the full student, the six half students and
# the 12 merges. Read with ../evaluate/merge_report.py --task <task>_kdgold75.
set -euo pipefail
T=${1:?usage: run_gold_kd_merges.sh <billsum_split|cnndm|samsum>}
M4=google/gemma-3-4b-it
case $T in
  # max-tokens: the longest reference kept; seq: teacher prompt + that, and the
  # sequence_len of <task>_kd_gold.yaml. gpu: vLLM memory share, lower for long
  # prompts because prompt logprobs materialise full-vocabulary rows.
  billsum_split) MAXTOK=2048; SEQ=6144; GPU=0.5; MIN=();                 TT=03:00:00; ET=03:00:00; EV=(-T prompt_variant=short) ;;
  cnndm)         MAXTOK=384;  SEQ=3072; GPU=0.7; MIN=(--min-steps 150);  TT=01:30:00; ET=02:00:00; EV=() ;;
  samsum)        MAXTOK=128;  SEQ=2048; GPU=0.9; MIN=();                 TT=01:30:00; ET=01:00:00; EV=(-T prompt_variant=short) ;;
  *) echo "unknown task $T" >&2; exit 1 ;;
esac
P=$PROJECTDIR/$USER/mm/models
uv run --no-sync --project ../train python prepare_gold_kd_prompts.py "$T"
sub() { sbatch --parsable "$@"; }
ev() { (cd ../evaluate && sub --job-name="eval-$T-gold75-$1" --time=$ET ${2:+--dependency=afterok:$2} \
  evaluate.sbatch "$T" "${@:3}" "${EV[@]}"); }

TE=$(sub --job-name="kd-teacher-gold75-$T" --time=01:30:00 kd_screen.sbatch teacher ../datasets/${T}_kd_prompts ${T}_kdgold \
  --targets gold --max-tokens $MAXTOK --max-seq-len $SEQ --gpu-memory-utilization $GPU)
PR=$(sub --job-name="kd-prep-gold75-$T" --time=01:00:00 --dependency=afterok:$TE kd_screen.sbatch preprocess ${T}_kd_gold.yaml)
FU=$(sub --job-name="kd-gold75-$T-full" --time=$TT --dependency=afterok:$PR kd_screen.sbatch trainloop $M4 \
  ../models/gemma3-$T-kdgold75_full-lora ../datasets/${T}_kdgold_prepared --ce-weight 0.75 "${MIN[@]}")
SP=$(sub --job-name="kd-split-gold75-$T" --time=00:30:00 --dependency=afterok:$PR kd_screen.sbatch split \
  ../datasets/${T}_kdgold_prepared ../datasets/${T}_kdgold_half)
echo "teacher $TE, preprocess $PR, full $FU, split $SP"
echo "evals: base $(ev base "" $M4), full $(ev full $FU $M4 --adapter ../models/gemma3-$T-kdgold75_full-lora)"
for seed in 42 2 3; do
  s=$([[ $seed == 42 ]] && echo "" || echo "_s$seed")
  ids=()
  for h in a b; do
    A=../models/gemma3-$T-kdgold75_$h$s-lora
    J=$(sub --job-name="kd-gold75-$T-$h$s" --time=$TT --dependency=afterok:$SP kd_screen.sbatch trainloop $M4 $A \
      ../datasets/${T}_kdgold_half_$h --ce-weight 0.75 --seed $seed "${MIN[@]}")
    ids+=("$J"); echo "half $h$s: train $J, eval $(ev $h$s $J $M4 --adapter $A)"
  done
  for m in linear ties dare_ties task_arithmetic; do
    J=$(cd ../merge && sub --job-name="merge-$T-gold75-$m$s" --dependency=afterok:${ids[0]}:${ids[1]} lora_merge.sbatch \
      $T-kdgold75-merged-$m$s $m ../models/gemma3-$T-kdgold75_a$s-lora ../models/gemma3-$T-kdgold75_b$s-lora)
    echo "merge $m$s: $J, eval $(ev merged-$m$s $J $P/$T-kdgold75-merged-$m$s)"
  done
done
