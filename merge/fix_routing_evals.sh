#!/bin/bash
# Repairs the provenance of the fitted and trained routers' MoEs and resubmits
# their failed evaluations. Run from $PROJECTDIR/$USER/mm-moe/merge.
set -euo pipefail
export HF_HUB_CACHE=$PROJECTDIR/$USER/mm/hf-cache HF_TOKEN_PATH=$HOME/.cache/huggingface/token UV_NO_SYNC=1
patch() {
python3 - "$@" <<'PY'
import json, sys
from pathlib import Path
for d in map(Path, sys.argv[1:]):
    p, base = d / "provenance.json", d.parent / "mistral_7b_task_moe_4" / "provenance.json"
    if not p.exists() or "coverage" in json.loads(p.read_text()):
        continue
    old = json.loads(p.read_text())
    rec = json.loads(base.read_text())
    rec["routing"] = {"made_by": "merge/moe_routing.py", **old.get("routing", {})}
    p.write_text(json.dumps(rec, indent=1, default=str))
    print("patched", d)
PY
}
patch ../models/task_merge/seed_*/mistral_7b_task_moe_4_{fit,train}
for S in 1 2; do for v in fit train; do
  build=moe_${v}_mistral_s$S
  state=$(sacct -u $USER -S 2026-11-05T15:30 -X -n -P --name $build -o State | tail -1)
  dep=()
  if [[ $state != COMPLETED ]]; then  # still to finish: patch once it has, before its evaluations
    b=$(squeue --me -h -n $build -o %i | head -1); [[ -n $b ]] || { echo "$build: $state"; continue; }
    p=$(sbatch --parsable --job-name=patch_${v}_s$S --dependency=afterok:$b --kill-on-invalid-dep=yes \
          --time=00:05:00 --wrap "cd $PWD && bash -c '$(declare -f patch); patch ../models/task_merge/seed_$S/mistral_7b_task_moe_4_$v'")
    dep=(--dependency=afterok:$p --kill-on-invalid-dep=yes)
  fi
  for t in crime wine "clinc150 -T prompt_variant=labels" "xsum -T prompt_variant=long -M batch_size=2"; do
    read -r task extra <<<"$t"; name=eval_moe_${task}_${v}_mistral_s$S
    st=$(sacct -u $USER -S 2026-11-05T15:30 -X -n -P --name $name -o State | tail -1)
    [[ $st == COMPLETED || $st == RUNNING ]] && continue
    scancel -n $name 2>/dev/null || true  # a pending one would start before the patch
    # shellcheck disable=SC2086
    (cd ../evaluate && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 sbatch --parsable \
      --job-name=$name ${dep[@]+"${dep[@]}"} --time=03:00:00 evaluate.sbatch $task \
      ../models/task_merge/seed_$S/mistral_7b_task_moe_4_$v $extra) >> ../evaluate/slurm/routing_jobs.txt
  done
done; done
tail -n +1 ../evaluate/slurm/routing_jobs.txt | wc -l
