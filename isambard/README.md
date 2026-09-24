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

Account `brics.u6ui`, partition `workq` (the only one, and the default). Both are
now set in `config.sh` and in the `#SBATCH` directives, so nothing needs editing.
Re-derive them on a different allocation with:

```bash
sacctmgr show assoc user=$USER format=account
sinfo -s
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
git clone https://github.com/alan-turing-institute/model-merging.git
cd model-merging && git checkout xsum-summarisation
bash isambard/00_setup_env.sh
bash isambard/01_prefetch.sh

# build the 8-way split and the configs it needs
XSUM_SPLITS=8 uv run --project evaluate python prepare_xsum_data.py
python isambard/make_shard_configs.py --splits 8     # add --no-4bit if needed
python isambard/make_shard_merge.py --splits 8       # the 8-way merge
python isambard/make_pair_merges.py --splits 8       # the four disjoint pairs
python isambard/make_kd_configs.py  --splits 8       # the three KD alpha arms

# the whole chain, each stage gated on the one before
TRAIN=$(sbatch --parsable isambard/10_train_shards.slurm)
MERGE=$(sbatch --parsable --dependency=afterok:$TRAIN isambard/20_merge.slurm)
sbatch --dependency=afterok:$MERGE isambard/21_eval.slurm

# the KD arm; teachers need only the shard experts, so it forks off $TRAIN
TEACH=$(sbatch --parsable --dependency=afterok:$TRAIN isambard/40_kd8_teachers.slurm)
CONCAT=$(sbatch --parsable --dependency=afterok:$TEACH isambard/41_kd8_concat.slurm)
sbatch --dependency=afterok:$CONCAT,afterok:$MERGE isambard/42_kd8_sweep.slurm
```

`prepare_xsum_data.py` writes `xsum_dataset1..N` for every N, so a run at a
different `XSUM_SPLITS` overwrites the previous one in place. It now records the
split count in each directory and refuses to overwrite one built differently
(`XSUM_OVERWRITE=1` to mean it). That matters on a machine that already holds
the two-way halves: without the guard, `XSUM_SPLITS=8` replaces half 1 with a
1/8 shard under the same path, and every adapter and KD number measured against
that path silently changes meaning.

## What the stages are, and why they are split this way

| stage | shape | what it is |
| --- | --- | --- |
| `10_train_shards` | array 1-8 | one shard expert per task, ~1,000 examples each |
| `20_merge` | array 1-5 | the 8-way merge, then the four disjoint pair merges |
| `21_eval` | array 1-14 | 8-way merge, 4 pairs, 8 shards, full - one per task |
| `40_kd8_teachers` | array 1-8 | teacher logprobs, expert i over shard i |
| `41_kd8_concat` | single | combines the eight into the KD training set |
| `42_kd8_sweep` | array 1-3 | 0.9/0.1, 0.5/0.5, 0.2/1.0 on one student |

Two of those splits are the point rather than tidiness. **The teacher pass is
produced once and consumed by all three alpha arms** - it is the most expensive
step in the chain, and the three arms are supposed to differ in two scalars, so
recomputing it per arm would both waste the GPU-hours and let the arms drift.
**Evaluation is an array rather than a loop**: fourteen evaluations back to back
inside one allocation is the same GPU-hours on a critical path fourteen times
longer, and a failure in the twelfth used to discard the eleven before it.

The concat is its own job because three sweep tasks start at once and would
otherwise race to write the same directory.

## The pair merges

Between one shard (1/8 of the data) and the 8-way merge there is a rung nobody
has measured: two shards merged, i.e. 1/4 of the data reached by averaging
rather than by training on it. The pairs are disjoint - (1,2), (3,4), (5,6),
(7,8) - so they partition the training set exactly once and read as four
independent draws of the same quantity. All 28 pairs would give a variance
estimate at seven times the cost and without that independence.

## The KD alpha sweep

`KD.md`'s 2026-09-24 section closes with what the alignment fix leaves open. The
0.9/0.1 -> 0.2/1.0 improvement was real, but the explanation on record - "the
weighting was the whole of it" - was written about a teacher that was scoring
the token after the one it described. Whether a *correctly aligned* teacher at
0.9/0.1 still destroys termination is untested, and it is the one thing three
arms on a shared student and shared teacher data can settle.

Read the result against the degeneracy counters, not only ROUGE: the pre-fix
0.9/0.1 arms returned 40-64 empty outputs and 54-74 with a token repeated three
or more times out of 1000, where every non-KD model produced zero of both.

```bash
uv run --project evaluate python experiments/bootstrap_rouge.py $RESULTS_DIR
uv run --project evaluate python experiments/degeneracy.py $RESULTS_DIR
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

## The KD re-run (`30_kd_xsum.slurm`)

Now the priority, because a0d3 — the Azure instance — is unavailable and because
every KD number on record was produced with a misaligned teacher (see the
2026-09-24 section of `KD.md`). This job re-runs the XSum arm with the corrected
producer.

It deliberately does **not** use `run_kd_pipeline.sh`: that script resolves every
artefact through the Azure ML registry and writes to `~/cloudfiles`, neither of
which exists here, so the registry bookkeeping would have to be stubbed rather
than ported. The four steps are written out directly — merge, teacher logprobs
per half, distil, evaluate — which is also easier to audit, and auditing is the
point of this run.

The first thing it does is run the teacher==student KL check and **exit non-zero
unless it reads 0.0000**. Everything after that point trains silently on noise if
the alignment is wrong, which is exactly how the last month of numbers were
generated.

```bash
# from the laptop, once: ~230 MB of adapters
bash isambard/02_push_adapters.sh      # uses the clifton alias u6ui.aip2.isambard

# on the cluster
sbatch isambard/30_kd_xsum.slurm
```

The adapters are copied rather than retrained on purpose. They are the exact
artefacts every number in `KD.md` was measured against; retraining them here would
change the experts as well as the alignment, and the re-run has to isolate one
variable. Only the adapter payload moves — `optimizer.pt` and the nested
checkpoint directory are ~290 MB each of training state that nothing reads.

Pre-fix baselines to compare against: KD 0.3987, merge 0.3970, better half 0.3990,
full-data 0.4049, and zero on all four degeneracy counters.

## Untested

None of this has run. I have no Isambard access from here, so the Slurm
directives, module names and partition are written from the documented setup, not
from a successful submission. Expect the first `sbatch` to need adjusting.
