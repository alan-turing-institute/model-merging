#!/bin/bash
# LLM-AggreFact (summarisation subsets) for DeepEval and the LLM judge, from evaluate/:
#   ./run_aggrefact.sh
# 1. prepare_aggrefact.py here (needs the gated dataset's terms accepted);
# 2. one job replaying the four (subset, split) claim sets as Inspect logs;
# 3. per subset, one summ_metrics job running the judge and DeepEval;
# then: uv run --no-sync python aggrefact_eval.py --md ../results/aggrefact.md
set -euo pipefail
OUT=../results/summ_metrics_aggrefact
X=${EXCLUDE_NODES:-nid010005,nid010008,nid010038}
(cd ../train && uv run --no-sync python ../evaluate/prepare_aggrefact.py ../datasets/aggrefact)
R=$(sbatch --parsable --exclude=$X --job-name=aggrefact-replay --time=00:30:00 --gres=gpu:1 --cpus-per-task=4 --mem=16000M \
  --wrap "export UV_NO_SYNC=1; for t in cnn xsum; do for s in dev test; do uv run inspect eval aggrefact_task.py@aggrefact_\$t -T split=\$s --model mockllm/model --log-dir logs --display none || exit 1; done; done")
echo "replay $R"
for t in cnn xsum; do
  J=$(sbatch --parsable --exclude=$X --time=06:00:00 --job-name=aggrefact-$t --dependency=afterok:$R summ_metrics.sbatch \
    --task aggrefact_$t --models "^aggrefact_$t-(dev|test)$" --metrics judge deepeval --concurrency 64 --out $OUT)
  echo "judge + deepeval $t: $J"
done
