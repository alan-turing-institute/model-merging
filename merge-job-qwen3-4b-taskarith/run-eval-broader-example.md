# Running the eval-broader experiment

Example walkthrough for someone starting fresh, cloning the repo for the
first time.

```bash
git clone https://github.com/alan-turing-institute/model-merging.git
cd model-merging/merge-job-qwen3-4b-taskarith
```

Submit the job (assumes `az` with the `ml` extension installed and logged
into an account with a role on `TIRE-1`/`TIRE-2`):

```bash
az ml job create -f job-eval-broader.yml --resource-group TIRE-1 --workspace-name TIRE-2
```

That prints a job name (something like `frosty_lemon_abc123xyz`). Stream logs live:

```bash
az ml job stream --name <job-name> --resource-group TIRE-1 --workspace-name TIRE-2
```

Or just poll status without the full log firehose:

```bash
az ml job show --name <job-name> --resource-group TIRE-1 --workspace-name TIRE-2 --query status -o tsv
```

Once it finishes, pull the results down:

```bash
az ml job download --name <job-name> --resource-group TIRE-1 --workspace-name TIRE-2 -o results-broader
```

`results-broader/lm-eval/` will then have per-model subfolders for all 12
models (8 source checkpoints + linear/ties/dare-ties/task-arithmetic
merges), each scored on `arc_easy,piqa,hellaswag,mmlu`.
