#!/bin/bash
# Submits the routing experiments for the rebuilt Mistral MoE, seeds given (default 1 2),
# waiting on the rebuild's jobs by name. Run from $PROJECTDIR/$USER/mm-moe/merge.
set -euo pipefail
SEEDS=("${@:-1 2}"); SEEDS=(${SEEDS[*]})
export HF_HUB_CACHE=$PROJECTDIR/$USER/mm/hf-cache HF_TOKEN_PATH=$HOME/.cache/huggingface/token UV_NO_SYNC=1
B=mistralai/Mistral-7B-Instruct-v0.3
MAP=../evaluate/slurm/routing_jobs.txt
id() { squeue --me -h -n "$1" -o %i | head -1; }
# deps <job name...>: the queued ones as an afterok list, or - if all have finished
deps() { local ids=(); for n in "$@"; do j=$(id "$n"); [[ -n $j ]] && ids+=("$j"); done
         ((${#ids[@]})) && (IFS=:; echo "${ids[*]}") || echo -; }
sub() {  # sub <dir> <name> <deps or -> <sbatch args...>
  local dir=$1 name=$2 dep=$3; shift 3
  local args=(--parsable --job-name="$name")
  [[ $dep != - ]] && args+=(--dependency=afterok:$dep --kill-on-invalid-dep=yes)
  local j; j=$(cd "$dir" && sbatch "${args[@]}" "$@"); echo "$j $dep $name" >> $MAP; echo "$j"
}
evals() {  # evals <build job> <model dir under models/> <tag>
  for t in crime wine "clinc150 -T prompt_variant=labels" "xsum -T prompt_variant=long -M batch_size=1"; do
    set -- "$1" "$2" "$3"; read -r task extra <<<"$t"
    # shellcheck disable=SC2086
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
      sub ../evaluate eval_moe_${task}_${3} "$1" --time=04:00:00 evaluate.sbatch $task ../models/$2 $extra >/dev/null
  done
}
for S in "${SEEDS[@]}"; do
  M=task_merge/seed_$S/mistral_7b_task_moe_4
  base=$(deps moe_mistral_s$S)
  experts=$(deps convert_crime_mistral_s$S convert_wine_mistral_s$S convert_clinc150_mistral_s$S mlp_xsum_mistral_s$S)
  sub . probe_mistral_s$S "$base" moe_routing.sbatch probe ../models/$M ../models/$M/probe.json >/dev/null
  # option 1: real chat-formatted examples as mergekit's prompts
  ../evaluate/.venv/bin/python moe_routing.py prompts moe/mistral_task_moe_4_seed_${S}_config.yaml \
    moe/mistral_task_moe_4_seed_${S}_prompts_config.yaml --seed $S
  j=$(sub . moe_prompts_mistral_s$S "$experts" moe.sbatch moe/mistral_task_moe_4_seed_${S}_prompts_config.yaml \
    ../models/${M}_prompts --seed $S)
  evals "$j" ${M}_prompts prompts_mistral_s$S
  sub . probe_prompts_mistral_s$S "$j" moe_routing.sbatch probe ../models/${M}_prompts ../models/${M}_prompts/probe.json >/dev/null
  # option 2: routers fitted on the base model's hidden states
  j=$(sub . moe_fit_mistral_s$S "$base" moe_routing.sbatch fit ../models/$M ../models/${M}_fit --base $B --seed $S)
  evals "$j" ${M}_fit fit_mistral_s$S
  sub . probe_fit_mistral_s$S "$j" moe_routing.sbatch probe ../models/${M}_fit ../models/${M}_fit/probe.json >/dev/null
  # option 3: routers trained on the MoE, routing and answer losses
  j=$(sub . moe_train_mistral_s$S "$base" --time=06:00:00 moe_routing.sbatch train ../models/$M ../models/${M}_train --seed $S)
  evals "$j" ${M}_train train_mistral_s$S
  sub . probe_train_mistral_s$S "$j" moe_routing.sbatch probe ../models/${M}_train ../models/${M}_train/probe.json >/dev/null
done
wc -l < $MAP
