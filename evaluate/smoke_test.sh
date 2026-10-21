#!/bin/bash
# End-to-end smoke test of the Inspect tasks - no GPU, no gated model, no
# prepared data. Inspect's built-in mockllm provider stands in for the model and
# a three-row synthetic split stands in for each dataset, so this runs on a
# laptop or a CI runner in a few minutes.
#
# It checks that each task loads, builds its prompts with the training-side
# prompt function, generates, scores and writes a successful .eval log, and
# that the hf-peft provider and the faithfulness scorer import. Those are the
# failures that otherwise surface only when a coworker first tries to run
# something. It does NOT check that any score is right, and it does not load
# the faithfulness models (about 6 GB).
#
#   cd evaluate && bash smoke_test.sh
#   SMOKE_DIR=/tmp/smoke bash smoke_test.sh      # keep data and logs there
set -euo pipefail
cd "$(dirname "$0")"
WORK=${SMOKE_DIR:-$(mktemp -d)}
mkdir -p "$WORK"
echo "smoke test working directory: $WORK"

# --frozen: install exactly what uv.lock says and never rewrite it, as a
# coworker's first run would. A pyproject.toml that has drifted from the lock
# fails here rather than being silently re-resolved.
uv sync --frozen

uv run --frozen python - "$WORK" <<'PY'
import sys
from datasets import Dataset
work = sys.argv[1]
# Synthetic rows, written for this test: no licence question, and they name
# people and numbers so the entity-grounding scorer has something to check.
Dataset.from_list([
    {"id": "s1", "document": "Maria Lopez, 42, opened a bakery in Leeds on Monday.\nShe hired 6 staff.",
     "summary": "A Leeds woman has opened a bakery and hired six people."},
    {"id": "s2", "document": "The council approved 120 new homes after a two-hour debate on Tuesday.",
     "summary": "Plans for 120 homes have been approved by the council."},
    {"id": "s3", "document": "Heavy rain closed the A64 near York for five hours, police said.",
     "summary": "Flooding shut a road near York for several hours."},
]).save_to_disk(f"{work}/xsum")
Dataset.from_list([
    {"posts_name": "Someone broke into my car last night and took my laptop.", "label": 1},
    {"posts_name": "What is the best way to learn to bake sourdough at home?", "label": 0},
    {"posts_name": "My neighbour keeps parking across my driveway. Any advice?", "label": 0},
]).save_to_disk(f"{work}/crime")
PY

run() { uv run --frozen inspect eval "$@" --model mockllm/model --log-dir "$WORK/logs"; }
run xsum_task.py  -T prompt_variant=short -T dataset_path="$WORK/xsum"
run xsum_task.py  -T prompt_variant=long  -T dataset_path="$WORK/xsum"
run crime_task.py -T dataset_path="$WORK/crime"

uv run --frozen python -c "import hf_peft_provider, xsum_faithfulness; print('hf_peft_provider and xsum_faithfulness import')"

uv run --frozen python - "$WORK/logs" <<'PY'
import sys
from inspect_ai.log import list_eval_logs, read_eval_log
logs = [read_eval_log(info) for info in list_eval_logs(sys.argv[1])]
problems = []
for log in logs:
    errors = [s.id for s in (log.samples or []) if s.error]
    scored = [s for s in (log.samples or []) if s.scores]
    name = f"{log.eval.task} {log.eval.task_args}"
    if log.status != "success": problems.append(f"{name}: status {log.status}")
    if errors: problems.append(f"{name}: sample errors {errors}")
    if len(scored) != 3: problems.append(f"{name}: {len(scored)} of 3 samples scored")
    print(f"  {log.status:8} {name}")
if len(logs) != 3: problems.append(f"expected 3 logs, found {len(logs)}")
if problems:
    sys.exit("SMOKE TEST FAILED:\n  " + "\n  ".join(problems))
print("smoke test passed")
PY
