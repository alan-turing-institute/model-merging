# Shared settings for the Isambard-AI port. Source this from every script.
#
# FILL THESE IN - they are the only site-specific values, and nothing runs
# until they are right.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-brics.u6ui}"    # sacctmgr show assoc user=$USER format=account
SLURM_PARTITION="${SLURM_PARTITION:-workq}"     # sinfo -s (workq is the only one, and default)
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"               # a GH200 node exposes 4

# Everything below derives from those, or from where you cloned the repo.
REPO_ROOT="${REPO_ROOT:-$HOME/model-merging}"

# Compute nodes have no outbound network on Isambard-AI, so every weight and
# dataset has to be resident before a job starts. Point the Hugging Face cache
# at project space and pre-fetch on the login node (01_prefetch.sh).
#
# HF_HUB_CACHE, not HF_HOME: HF_HOME moves the auth token as well as the cache,
# which silently logs you out and makes gated models 401 halfway through a run.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/.hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"    # set to 0 on the login node
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"

# Results go to project space, not $HOME - $HOME is small and not mounted the
# same way on every node.
export RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results-isambard}"

# The pipeline scripts were written against the Azure ML registry. On Isambard
# there is no registry, so every artefact is a path and version pinning is by
# directory name.
export MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
export DATASETS_DIR="${DATASETS_DIR:-$REPO_ROOT/../datasets}"

# Compute nodes have no outbound network, and axolotl's telemetry retries
# against posthog.com regardless - the ctx arm logged connection errors and then
# sat in training for five hours before the wall clock killed it, against twenty
# minutes for the same step on the shared-prompt arm. Whether the retries were
# the whole cause or not, telemetry from an air-gapped node can only cost time.
export DO_NOT_TRACK=1
export AXOLOTL_DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
