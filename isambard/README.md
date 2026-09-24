# Running the XSum arm on Isambard-AI

Why move: the halves design saturated. 4,000 XSum examples already reach 0.3990
ROUGE-1 against full-data's 0.4049, so the recovery fraction normalises by
0.0059 — a gap a paired bootstrap cannot resolve at n = 1000 (p = 0.058). The
fix is smaller shards, which make weaker experts and so a larger denominator,
and that means eight training runs instead of two. An eight-task Slurm array is
the natural shape for it, and Isambard-AI's GH200s are both faster and less
contended than one shared A100 instance.

Expected: 1,000-example shards land near 0.36, putting the gap to full-data at
roughly 0.04 — about seven times the current denominator.

## Before anything runs

Two values are site-specific and nothing works until they are right. Put them in
your environment or edit `config.sh`:

```bash
sacctmgr show assoc user=$USER format=account   # -> SLURM_ACCOUNT
sinfo -s                                        # -> SLURM_PARTITION
```

## The architecture risk, which is the real gate

Isambard-AI is **GH200: aarch64 CPUs with Hopper GPUs**. Several of axolotl's
dependencies ship x86-only wheels:

| package | situation |
|---|---|
| `flash-attn` | no aarch64 wheel; source build, 45–90 min, needs nvcc |
| `bitsandbytes` | aarch64 supported from 0.43, but the usual failure point |
| `xformers` | frequently resolves to a source build |

`00_setup_env.sh` reports which of these imported, so find out before queueing
anything. **If `bitsandbytes` will not build**, the training configs' `load_in_4bit:
true` cannot work. The fallback is plain LoRA in bf16 — a GH200 has 96 GB of HBM
against the A100's 40, so a 4B model with LoRA fits easily and 4-bit quantisation
is not needed for capacity here. Generate the configs with `--no-4bit`, and
record it in `PREREGISTRATION.md` as a deviation: it changes the numbers slightly
and must not be swapped in silently.

Compute nodes have **no outbound network**, so `01_prefetch.sh` runs on a login
node and must complete before any job is submitted. `gemma-3-4b-it` is gated —
`hf auth login` first. Note `config.sh` sets `HF_HUB_CACHE`, not `HF_HOME`:
redirecting `HF_HOME` moves the auth token as well as the cache, which silently
logs you out and 401s a gated model halfway through a run. That cost a day on
Azure.

## Sequence

```bash
# login node, once
bash isambard/00_setup_env.sh
bash isambard/01_prefetch.sh

# build the 8-way split and the configs it needs
XSUM_SPLITS=8 uv run --project evaluate python prepare_xsum_data.py
python isambard/make_shard_configs.py --splits 8     # add --no-4bit if needed
python isambard/make_shard_merge.py --splits 8

# train the eight experts, then merge and evaluate when they all succeed
JOB=$(sbatch --parsable isambard/10_train_shards.slurm)
sbatch --dependency=afterok:$JOB isambard/20_merge_eval.slurm
```

`prepare_xsum_data.py` keeps `XSUM_SPLITS=2` on the original `train_test_split`
construction rather than a contiguous shard, because every half-expert, merge and
result on record was trained against that membership. Only `SPLITS > 2` uses
contiguous shards.

## Two design decisions worth not re-litigating

**Linear at 1/N, not task-arithmetic.** Eight shard experts all point the same
direction. The crowding curve measured that what degrades a merge is total
displacement per direction, not adapter count: three same-direction adapters
under task-arithmetic at weight 1.0 dropped crime accuracy to 0.5348, below the
untuned base model. Averaging holds displacement at roughly one expert's worth.

**Pre-specify the baseline shard.** *R* normalises by the best expert, and the
maximum of eight noisy estimates is biased upward — a winner's curse that deflates
*R*. Use shard 1, or the mean of the eight, and state which before looking.

## What is not ported

The KD arm. `run_kd_pipeline.sh` resolves artefacts through the Azure ML registry
and writes to `~/cloudfiles`; on Isambard there is no registry and every artefact
is a path. Porting it means replacing `pipeline_resolve_results_dir` and the
`recorded_version` bookkeeping with directory-name pinning. Worth doing only once
the shard design is shown to open a usable gap — otherwise it ports a pipeline for
an experiment that still cannot answer anything.

## Untested

None of this has run. I have no Isambard access from here, so the Slurm
directives, module names and partition are written from the documented setup, not
from a successful submission. Expect the first `sbatch` to need adjusting.
