#!/bin/bash
# Rebuilds main's Mistral task-merging MoE (seeds 1 and 2) in its own worktree,
# $PROJECTDIR/$USER/mm-moe at origin/main: the XSum teacher (once), the four
# MLP experts, the conversions to full models, base and expert evaluations, and
# merge/submit_moe.sh (MoE + 4 evals). Job ids go to evaluate/slurm/rebuild_jobs.txt.
set -euo pipefail
P=$PROJECTDIR/$USER
W=$P/mm-moe
B=mistralai/Mistral-7B-Instruct-v0.3
export HF_HUB_CACHE=$P/mm/hf-cache HF_TOKEN_PATH=$HOME/.cache/huggingface/token UV_NO_SYNC=1

if [[ ! -d $W ]]; then
  git -C ~/model-merging fetch -q origin
  git -C ~/model-merging worktree add -q --detach "$W" origin/main
  ln -sfn $P/mm/venvs/dp1b-evaluate "$W/evaluate/.venv"
  ln -sfn ~/model-merging/merge/.venv "$W/merge/.venv"
  ln -sfn $P/mm/venvs/self-distill "$W/self-distill/.venv"
  ln -sfn ~/model-merging/train/.venv "$W/train/.venv"
  rm -rf "$W/datasets" && ln -s $P/model-merging/datasets "$W/datasets"
  mkdir -p "$W/models" "$W/evaluate/slurm" "$W/evaluate/logs"
fi
cd "$W"
git log -1 --format='worktree at %h %s'
MAP=$W/evaluate/slurm/rebuild_jobs.txt
sub() {  # sub <dir> <name> <dependency or -> <sbatch args...>
  local dir=$1 name=$2 dep=$3; shift 3
  local args=(--parsable --job-name="$name")
  [[ $dep != - ]] && args+=(--dependency=afterok:$dep --kill-on-invalid-dep=yes)
  local j; j=$(cd "$dir" && sbatch "${args[@]}" "$@")
  echo "$j $dep $name" >> "$MAP"; echo "$j"
}

teacher=$(sub self-distill teacher_xsum_mistral - generate_teacher_logprobs.sbatch "$B" \
  ../datasets/xsum/xsum_prompts ../datasets/xsum mistral_7b_xsum_gold_kd_data --targets gold --seed 1)
sub evaluate eval_base_crime_mistral - --time=02:00:00 evaluate.sbatch crime "$B" >/dev/null
sub evaluate eval_base_wine_mistral - --time=02:00:00 evaluate.sbatch wine "$B" >/dev/null
sub evaluate eval_base_clinc150_mistral - --time=03:00:00 evaluate.sbatch clinc150 "$B" -T prompt_variant=labels >/dev/null
sub evaluate eval_base_xsum_mistral - --time=03:00:00 evaluate.sbatch xsum "$B" -T prompt_variant=long >/dev/null

for S in 1 2; do
  for spec in "crime crime_dataset mistral_7b_crime_mlp 02:00:00" \
              "wine wine_dataset mistral_7b_wine_mlp 02:00:00" \
              "clinc150 clinc150_labels_dataset mistral_7b_clinc150_labels_mlp 03:00:00"; do
    read -r task data name etime <<<"$spec"
    out=models/$task/seed_$S/$name
    extra=(); [[ $task == clinc150 ]] && extra=(-T prompt_variant=labels)
    t=$(sub train mlp_${task}_mistral_s$S - train.sbatch MLP_${task}_mistral.yaml \
      ../datasets/$task/$data ../$out --seed $S)
    sub . convert_${task}_mistral_s$S "$t" convert_to_full_model.sbatch "$B" $out ${out}_full >/dev/null
    sub evaluate eval_mlp_${task}_mistral_s$S "$t" --time=$etime evaluate.sbatch $task "$B" --adapter ../$out \
      ${extra[@]+"${extra[@]}"} >/dev/null
  done
  x=$(sub self-distill mlp_xsum_mistral_s$S "$teacher" train_kd.sbatch MLP_xsum_mistral.yaml \
    ../models/xsum/seed_$S/mistral_7b_xsum_gold_long_mlp ../datasets/xsum/mistral_7b_xsum_gold_kd_data_long --seed $S)
  sub evaluate eval_mlp_xsum_mistral_s$S "$x" --time=03:00:00 evaluate.sbatch xsum \
    ../models/xsum/seed_$S/mistral_7b_xsum_gold_long_mlp -T prompt_variant=long >/dev/null
  (cd merge && bash submit_moe.sh mistral $S --submit | tail -1)
done
wc -l < "$MAP"
