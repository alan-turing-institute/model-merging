#!/bin/bash
# SummEval comparison (Fabbri et al., 2021), from evaluate/:
#   ./run_summeval.sh
# 1. prepare_summeval.py (here, on the login node): the 100 annotated articles,
#    their 16 rated systems' summaries and expert ratings.
# 2. one job replaying the 16 systems and the reference as Inspect logs;
#    our CNN/DailyMail models generating on the same 100 articles.
# 3. summ_metrics.sbatch on both: the LLM judge, BERTScore (ours and the
#    toolkit's configuration) and the SummEval toolkit metrics, into
#    ../results/summ_metrics_summeval.
# Then: uv run python summeval_meta.py --md ../results/summeval-comparison.md
set -euo pipefail
OUT=../results/summ_metrics_summeval
DS=../datasets/summeval
M4=google/gemma-3-4b-it
P=$PROJECTDIR/$USER/mm/models
(cd ../train && uv run --no-sync python ../evaluate/prepare_summeval.py "$DS")

systems=$(cd ../train && uv run --no-sync python -c "from datasets import load_from_disk; print(' '.join(sorted(set(load_from_disk('$DS/pairs')['system']))))")
# Sized like the other small jobs: with no --gres the partition left it pending on
# Resources behind whole-node allocations.
R=$(sbatch --parsable --job-name=summeval-replay --time=01:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=16000M --wrap "export UV_NO_SYNC=1; for s in $systems; do uv run inspect eval summeval_task.py -T system=\$s --model mockllm/model --log-dir logs --display none || exit 1; done")
echo "replay $R ($systems)"

ids=()
ev() { local name=$1; shift; ids+=("$(sbatch --parsable --job-name="eval-summeval-$name" --time=01:00:00 evaluate.sbatch cnndm "$@" -T dataset_path=$DS/cnndm_test)"); }
ev base $M4
ev sft500 $M4 --adapter ../models/gemma3-cnndm_curve-n500-lora
ev sft8000 $M4 --adapter ../models/gemma3-cnndm_curve-n8000-lora
ev kdfull $M4 --adapter ../models/gemma3-cnndm-kdgold75_full-lora
for s in "" _s2 _s3; do
  for h in a b; do ev "kd$h$s" $M4 --adapter ../models/gemma3-cnndm-kdgold75_$h$s-lora; done
  for m in linear ties dare_ties task_arithmetic; do ev "merged-$m$s" $P/cnndm-kdgold75-merged-$m$s; done
done
echo "evals: ${ids[*]}"

A=$(sbatch --parsable --job-name=judge-summeval --time=08:00:00 --dependency=afterok:$R summ_metrics.sbatch \
  --task summeval --models "^summeval-" --metrics judge bertscore summeval --concurrency 64 --out $OUT)
B=$(sbatch --parsable --job-name=judge-summeval-ours --time=08:00:00 --dependency=afterok:$(IFS=:; echo "${ids[*]}") \
  summ_metrics.sbatch --task cnndm --dataset summeval/cnndm_test --metrics judge bertscore summeval --concurrency 64 --out $OUT \
  --models "^(base-gemma-3-4b-it|gemma3-cnndm_curve-n(500|8000)-lora|gemma3-cnndm-kdgold75_(full|[ab](_s2|_s3)?)-lora|cnndm-kdgold75-merged-(linear|ties|dare_ties|task_arithmetic)(_s2|_s3)?)$")
echo "metrics: systems $A, ours $B"
