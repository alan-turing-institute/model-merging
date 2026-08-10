# Running model-merging experiments on Azure ML — handover notes

Written from the MergeKit-paper reproduction (`merge-job-mergekit-paper-repro/`,
replicating arXiv 2403.13257 Table 1) as the worked example, since it exercised
almost every gotcha you'll hit. The same pattern was reused for the Qwen3-4B
n=7 linearity/new-methods work (`merge-job-qwen3-4b-taskarith/`) and the
Qwen2.5-0.5B/1.5B experiments.

## 1. Prerequisites

- `az` CLI with the `ml` extension installed, logged in, subscription set:

  ```bash
  brew install azure-cli
  # locked-down machine, no admin/sudo rights? use a venv instead of brew:
  #   python3 -m venv ~/azure-cli-venv && source ~/azure-cli-venv/bin/activate
  #   pip install azure-cli
  az extension add -n ml
  az login
  az account set --subscription "<name-or-id>"
  ```

- Resource group and workspace: **`--resource-group TIRE-1 --workspace-name TIRE-2`**
  on every `az ml` call (see `merge-job-parallel-template/submit_parallel.sh`) —
  **except** `az ml workspace show`, which takes `--name` instead of
  `--workspace-name` (it's the resource being shown, not a scope argument on
  a sub-resource). Verify access before anything else, since a failure here
  means the rest of this doc doesn't apply to you yet:

  ```bash
  az ml workspace show --name TIRE-2 --resource-group TIRE-1
  az ml compute show --name gpu-cluster-merge --resource-group TIRE-1 --workspace-name TIRE-2
  az ml compute show --name a100-cluster-merge --resource-group TIRE-1 --workspace-name TIRE-2
  ```

- An HF token with access to any gated models (Llama-2, etc.) — passed as an
  `environment_variables.HF_TOKEN` in the job YAML, not baked into the image.
- Two GPU compute targets exist: `azureml:gpu-cluster-merge` (T4, 16GB) and
  `azureml:a100-cluster-merge` (A100, 80GB). Which one to use is a real
  decision, not a default — see §5.

## 2. Anatomy of one experiment folder

Every `merge-job-*` folder follows the same shape:

```
merge-job-<name>/
├── docker-context/
│   └── Dockerfile          # base image + pinned deps (see §4 for why pins matter)
├── configs/
│   └── <method>-*.yml      # one MergeKit YAML per merge method/topology
├── run_*.sh                # the actual shell script the job container executes
├── job*.yml                # AML job spec(s): command, compute, inputs/outputs, env vars
└── compare_to_paper.py     # optional: diff our numbers against a published baseline
```

The job YAML (`job.yml`) is the AML-native description; the shell script it
invokes (`run_all.sh` / `run_dare_ties_fixed.sh` / etc.) is where the actual
`mergekit-yaml` and `lm_eval` commands live. Keep merge and eval logic in the
shell script, not inline in the YAML `command:` block — it's much easier to
patch a bug and resubmit without touching the AML spec.

Minimal job.yml shape (from `job.yml`):

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
display_name: mergekit-paper-repro-llama2-meditron
experiment_name: model-merge-experiments
code: .
command: >-
  bash run_all.sh ${{outputs.results}} ${{outputs.merged_models}}
environment:
  build:
    path: docker-context
environment_variables:
  HF_TOKEN: hf_...
compute: azureml:a100-cluster-merge
resources:
  instance_count: 1
limits:
  timeout: 21600
outputs:
  results:        { type: uri_folder, mode: upload }
  merged_models:  { type: uri_folder, mode: rw_mount }
```

Key output-mode distinction that caused a real disk-quota bug (§4):
- `mode: upload` — synced incrementally to blob storage as the job writes;
  survives even if the job later crashes.
- `mode: rw_mount` — mounted as a writable local path, counts against
  `AZ_BATCH_NODE_ROOT_DIR`'s disk quota (~64GB) just like any other local
  write. Don't `rw_mount` large model checkpoints unless you actually need
  random-access re-reads later.

## 3. Submitting and monitoring

```bash
az ml job create -f job.yml --resource-group TIRE-1 --workspace-name TIRE-2
az ml job show --name <job-name> --resource-group TIRE-1 --workspace-name TIRE-2 --query status -o tsv
az ml job stream --name <job-name> --resource-group TIRE-1 --workspace-name TIRE-2   # tail logs live
```

To fan work out across nodes instead of running everything in one sequential
job (used for the 7-model Qwen2.5 comparison, ~3-4x wall-clock speedup):
submit each merge/eval as its own `az ml job create ... --set inputs.xxx=...`
call, capture the job name with `--query name -o tsv`, then `az ml job wait
--name "$JOB"` before the dependent step. See
`merge-job-parallel-template/submit_parallel.sh` for the full pattern,
including chaining one job's output straight into the next job's input via
`--set inputs.model_dir.path="azureml://jobs/$PRIOR_JOB/outputs/merged_model"`.

**Registering an expensive intermediate as a reusable data asset** — do this
whenever a later step (e.g. re-running eval with a bug fix) shouldn't force
re-running an expensive merge:

```bash
az ml data create --name mergekit-repro-lerp --version 1 --type uri_folder \
  --path azureml://jobs/<merge-job-name>/outputs/merged_models \
  --resource-group TIRE-1 --workspace-name TIRE-2
```

Then reference it in a later job.yml as an input:

```yaml
inputs:
  merged_lerp:
    type: uri_folder
    path: azureml:mergekit-repro-lerp@latest
    mode: ro_mount
```

This is exactly how the LERP config-bug fix and the DARE-TIES dtype fix were
each re-evaluated without re-merging everything (`job-eval-lerp-fixed.yml`,
`job-eval-merged-only.yml`).

## 4. Gotchas actually hit (in the order they bit us)

Treat this as a checklist before assuming a failure is something new.

1. **`huggingface_hub>=1.16` breaks bare-slug dataset URIs.** Its tightened
   `hf://` parser requires `namespace/name` repo ids; legacy canonical
   datasets like `medmcqa` raise `HfUriError` before eval even starts.
   Fix: pin `huggingface_hub<1.16` in the Dockerfile.
2. **`datasets>=4.0` removed script-based dataset loaders entirely**, not just
   gated them behind `trust_remote_code`. `pubmedqa`'s underlying dataset
   (`bigbio/pubmed_qa`) still ships a builder script, so it hard-fails.
   Fix: pin `datasets<4.0.0`.
3. **`pubmedqa` still needs `trust_remote_code` even with the pin.**
   lm-eval-harness's own `--model_args` doesn't propagate this down to the
   dataset loader (known upstream issue, lm-eval-harness#2631).
   Fix: set `HF_DATASETS_TRUST_REMOTE_CODE: "1"` as a job env var — `datasets`'
   own opt-in mechanism, bypasses the harness entirely.
4. **Vocab-size mismatch on merge (e.g. Llama-2-7b-chat vocab=32000 vs
   Meditron-7b vocab=32017) breaks `transformers` loading** with
   `RuntimeError: ... ignore_mismatched_sizes=False`. Two fixes, not
   equivalent:
   - **Quick workaround**: `ignore_mismatched_sizes=True` in `model_args` —
     but this *fully discards and randomly reinitializes* the whole
     mismatched tensor (embed_tokens/lm_head), not just the extra rows.
     Fine for a "does this run at all" check, not for trusting the numbers.
   - **Real fix, used here**: inspect the actual tensor shapes directly
     (`safetensors.safe_open` on the merged checkpoint) — if the tensor data
     is healthy but the checkpoint's `config.json` just has the wrong
     `vocab_size` metadata (a real bug in this case, not corruption), patch
     `config.json` in blob storage directly and re-eval with no
     mismatch flag at all.
   - **The actually-correct long-term fix**, documented in the
     alan-turing-institute/model-merging repo's own `mergekit-handover.md`
     but never used in these configs: MergeKit's native
     `tokenizer_source: union` (or `base`) config key, which resolves vocab
     mismatches at merge time instead of leaving a landmine for eval.
5. **`dare_ties`'s default `rescale=True` can silently produce 100%-NaN
   embed/lm_head tensors in `float16`.** It divides the masked delta by the
   L1-norm of surviving (post-Bernoulli) elements; an unlucky mask draw can
   leave a near-zero norm, so the rescale factor overflows fp16's ~65504 max
   to `Inf` → `NaN`. Diagnose by downloading just the tensor (not the whole
   multi-GB checkpoint) via `az storage blob download --start-range/--end-range`
   using the safetensors header to compute the exact byte offset, and check
   with `torch.isnan(...).all()`. Fix: `dtype: bfloat16` (much larger dynamic
   range, same rescale behavior preserved — don't just disable rescale, that's
   no longer DARE-TIES).
6. **T4 (16GB) has a hard ceiling for DELLA / Model Breadcrumbs at n=7**,
   confirmed independently at two model scales (4B and 1.5B) — the bottleneck
   is holding all n stacked deltas through the sign-consensus/argsort step,
   and it scales with n, not just per-model size. No env var or dtype trick
   fixes this; it needs A100.
7. **`AZ_BATCH_NODE_ROOT_DIR` disk quota (~64GB) is invisible to `df -h /`**
   and counts `ro_mount` reads against the same budget as `rw_mount` writes —
   check `df -h` on the actual mount paths the job uses, not just `/`, when
   diagnosing disk-exhaustion failures.
8. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** only helps genuine
   memory *fragmentation* — it bought DELLA a bit more headroom before OOM but
   had zero effect on Breadcrumbs, which confirmed that one was a true
   shortfall, not fragmentation.

## 5. Compute placement: T4 vs A100 — don't default silently

T4 is cheaper and was the explicit default for most of this work; only move
to A100 when T4 has been shown to genuinely fail (OOM at more than one scale,
not just "it's slow"). The DELLA/Breadcrumbs n=7 case took this exact path:
T4 → T4 with `expandable_segments` → retried at a smaller model scale on T4 →
only then A100, once two independent T4 failures confirmed it wasn't fixable
in place. Respect whatever compute the user has specified for a given
experiment rather than upgrading to A100 by default "to be safe."

## 6. Comparing against a published baseline

`compare_to_paper.py` is the template for this: a `PAPER` dict of the
published numbers, a `PREFERRED_METRIC` map per task (`acc` vs `acc_norm` —
match whatever convention the source paper used, e.g. Open LLM Leaderboard
uses `acc_norm` for ARC-Challenge/HellaSwag but `acc` for MMLU/MedMCQA/etc.),
and a loader that globs `lm-eval`'s `results_*.json` output per model/task.
Run it against the downloaded `results` output folder:

```bash
az ml job download --name <eval-job-name> --resource-group TIRE-1 --workspace-name TIRE-2 -o results
python compare_to_paper.py results/lm-eval
```

## 7. Practical run order for a new repro-style experiment

0. First time submitting anything in this workspace, or unsure your access
   actually works end-to-end? Submit `merge-job-eval-smoketest/job.yml`
   as-is before writing anything new — it's an ungated model
   (`Qwen/Qwen2.5-0.5B`, no HF token needed), ~90min on T4, and confirms
   `az login`/workspace/compute access all actually work before you sink
   time into a new experiment folder or a multi-hour A100 job.
1. Write MergeKit configs (`configs/*.yml`) — one per method/topology.
2. Write `docker-context/Dockerfile` — start from a known-good base
   (`python:3.11-slim` + CUDA torch + `mergekit` + `lm-eval` + the three pins
   in §4 items 1-2 if any benchmark touches `medmcqa`/`pubmedqa`).
3. Write one `run_*.sh` that does: merge → eval baseline models → eval merged
   models, logging `df -h` before/after each merge (cheap insurance against
   gotcha #7).
4. Write `job.yml` pointing at that script, `compute:` picked per §5,
   `timeout:` generous (these runs are 2-6h on A100, more on T4).
5. Submit, stream logs, watch for the specific failure signatures in §4
   before assuming something new is wrong.
6. Once merges succeed but before evals do (or vice versa), register the good
   intermediate as a data asset (§3) so a bug fix in the other half doesn't
   force a full re-run.
7. `az ml job download` the `results` output, run `compare_to_paper.py` (or
   equivalent) against it.
