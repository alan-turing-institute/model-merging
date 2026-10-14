# Readme for using Axolotl to train models

Here we give details and code of how to train a model to be a classifier on crime data.

## Data preparation

First we use the `prepare_data.py` script to split the crime dataset into three parts, for training, validation and testing.  And also split the whole dataset in two halves before splitting into these three.

```bash
uv run python prepare_data.py
```

To run with other datasets etc, edit the python script.

## Axolotl for training models

The parameters for training using Axolotl are controlled via a [yaml file](crime_gemma.yaml).  For full details see the [online documentation](https://docs.axolotl.ai/docs/getting-started.html).

The key parameters are:

- `base_model` - this can be a HF address, or a path to a model on disk
- `datasets` - again can be a HF address, or a path to a dataset on disk.  You can also specify different training and validation datasets here.

Once you have a properly configured yaml, to train the model, simply run

```bash
uv run axolotl train crime_gemma.yaml
```

The LoRA adapter should then be saved in the Azure blob storage

```bash
az ml model create --name <name> --version 1 --type custom_model --path <local-path> --resource-group tire-1 --workspace-name tire-2
```

## BillSum: the data-scaling curve

XSum turned out to be saturated for this project's purposes — every arm, from
the untouched base model to the merges and the distilled students, landed inside
a band narrower than the seed-to-seed spread, so no merging question could be
answered on it. BillSum is the replacement candidate: US Congressional bills
(median 1,217 words against XSum's ~430) summarised into multi-sentence digests
(median 169 words against 21), so there is much more for a fine-tune to learn and
much more room between base and trained for a merge to fall into.

Before spending another forty arms on it, measure whether it has headroom at all.
That is what the scaling curve does: train the same configuration on ⅛, ¼, ½ and
all of the data, score each on the same held-out set, and look at whether the
score is still rising at the top of the curve.

Prepare the data **on a login node** — compute nodes have no outbound network:

```bash
uv run python prepare_billsum_data.py ../datasets/billsum
```

That writes four nested training arms (`full`, `half`, `quarter`, `eighth`), one
validation set shared by all of them, and two evaluation splits: `test` (federal
bills) and `ca_test` (California state bills, references roughly twice as long —
a genuine distribution shift rather than a second sample of the same thing).

The arms are **nested prefixes of one shuffled pool**, not four independent
draws: four draws would confound "less data" with "a different draw of data", and
the expert-draw standard deviation measured on the crime task (0.004–0.005) is
the same size as the effect being looked for.

Then submit the whole curve — four training arms, each evaluated as it finishes,
plus the untrained base model as the curve's n=0 point:

```bash
./run_billsum_curve.sh
```

and read it with:

```bash
cd ../evaluate && uv run python billsum_curve.py logs
```

That prints the four arms against their training-set sizes and runs a paired
bootstrap between consecutive arms. **If the half→full step is flat, BillSum is
saturated at this scale and the merging arms should not be run on it** — which is
a two-GPU-hour answer rather than a month-long one.

Two numbers in [`billsum_gemma.yaml`](billsum_gemma.yaml) and
[`prepare_billsum_data.py`](prepare_billsum_data.py) are one decision, not two:
`sequence_len: 4096` and the 2900-**token** document budget. Axolotl's default
`excess_length_strategy` is `drop`, so a budget raised without raising
`sequence_len` does not shorten the over-long examples — it deletes them,
shrinking the arms and plotting the curve against an x-axis that is wrong.

It does report the loss, in one `INFO` line among a long preprocessing log:
`Dropped N sequences outside valid range`. Measured on axolotl 0.18.0 with 10
rows of which 3 exceed the window:

| `excess_length_strategy` | rows kept | outcome |
| --- | --- | --- |
| `drop` (default) | 7 of 10 | logs `Dropped 3 sequences outside valid range` |
| `raise` | — | `ValueError`, job exits nonzero |
| `truncate` | 10 of 10 | cut to `sequence_len` |

The config sets **`raise`**, which turns an easily-missed log line into a failed
job.

Truncating by words instead of tokens is the obvious shortcut and it is a trap:
legislative text runs **1.74 tokens per word** (measured with the Gemma tokenizer
on this split) against ordinary prose's ~1.3, so a 2,400-word cap that a word-count
estimate put at ~3,700 tokens in fact produced examples of up to 6,948 and placed
7% of the full arm over the window. `prepare_billsum_data.py` now truncates in
tokens and verifies the fit on the rendered chat text, refusing to finish if any
example would be dropped; `train_billsum.sbatch` prints the row count again before
and after tokenisation.

## Headroom screens

Before spending merging runs on a dataset, a screen measures whether it has the
headroom a merge needs: how far training lifts the model above the untrained
base, and the denominator of the recovery fraction,
`R = (merged − best expert) / (full − best expert)`. XSum and SAMSum failed this
(the gain from more data is spent in the first few hundred examples); CLINC150
and EN–DE medical/legal passed.

### Before you start (Isambard-AI)

- **Sync environments once, on a login node, add-only:**
  `uv sync --frozen --inexact` in `train/` and `../evaluate/`. Jobs never sync
  (`UV_NO_SYNC=1`): concurrent array tasks race on one venv, and a plain (non
  `--inexact`) sync once removed a leftover cu12 wheel that shared files with the
  cu13 cuDNN, breaking torch in every venv.
- **Prepare data on a login node.** Compute nodes have no network.
- **Hugging Face:** weights live in `$PROJECTDIR/$USER/mm/hf-cache`
  (`HF_HUB_CACHE`, *not* `HF_HOME`, which also moves the token). Gemma and FLORES+
  are gated; log in with
  `ssh -t u6ui.aip2.isambard 'uvx --from huggingface_hub hf auth login --force'`.
- **`cuda_env.sh`** is sourced by the job scripts. It prepends the CUDA 13.0
  forward-compat libraries only when they are newer than the node driver: since
  the October 2026 driver upgrade (580.173.02) they are older, and loading them
  gives CUDA error 803 and no GPU. It also sets `UV_NO_SYNC`, `HF_HUB_CACHE` and
  turns off telemetry.

### The harness

| File | Role |
|---|---|
| `prepare_<task>_data.py` | builds `../datasets/<task>/`: one `<arm>/{train,validation}` per training arm, a shared `test` split, and an `ARMS` file listing the arms (largest first) |
| `<task>_gemma.yaml` | one axolotl config for every arm; the job overrides only paths and output |
| `train_screen.sbatch` | job array over the `ARMS` file: array index *i* trains arm *i* |
| `run_screen.sh <task>` | submits the training array, the untrained-model evaluation and one evaluation per arm, each gated on its own training task |
| `../evaluate/split_screen.py` | reads the logs: both denominators (random split and natural split) and the per-shard table |
| `../evaluate/scaling_curve.py` | the same for data-scaling curves (SAMSum, BillSum) |

```bash
cd train
uv run --no-sync python prepare_mt_domain_data.py ../datasets/mt_domain   # login node
./run_screen.sh mt_domain
cd ../evaluate && uv run python split_screen.py logs --task mt_domain --json ../results/mt_domain-screen.json
```

Two optional arguments rerun one arm (array index *i*) without touching the
original:

```bash
sbatch --array=3 train_screen.sbatch mt_domain 2                           # seed 2 -> expert_law_s2
sbatch --array=0 train_screen.sbatch cnndm "" google/gemma-3-1b-it          # 1B base -> full_1b
```

Pass these as arguments, never through `sbatch --export=ALL`: that carries the
login node's `TMPDIR` to compute nodes, where it does not exist. A 1B arm must be
evaluated with the 1B base: `evaluate.sbatch <task> google/gemma-3-1b-it --adapter ../models/gemma3-<task>-<arm>_1b-lora`.

### The screens

| Task | Dataset | Arms (training rows) | Headline metric | Result |
|---|---|---|---|---|
| `samsum` | SAMSum | n100 … n6400, nested (own scripts: `run_samsum_curve.sh`) | ROUGE-1 | flat from 100 rows — saturated |
| `billsum_split` | BillSum | full 1,000; two random halves of 500 | ROUGE-Lsum | denominator 0.026 (17× seed spread) |
| `bigpatent` | BigPatent, CPC sections c vs h | full, random half, one expert per section (2,000 each) | ROUGE-Lsum | 0.012 — sections transfer well |
| `cnndm` | CNN/DailyMail | full 4,000 | ROUGE-Lsum | base vs trained only |
| `xsum_screen` | XSum, articles cut to 512 words | full 4,000 | ROUGE-1 | base vs trained only |
| `clinc150` | CLINC150, 10 domains split 5/5 | full 15,000; random half; one expert per half | accuracy | 0.43 — mostly label coverage |
| `mt_domain` | EMEA vs JRC-Acquis, EN→DE | full 8,000; random half; one expert per domain | chrF++ | 2.44 (54× seed spread) |
| `mt_zh` | UN vs News Commentary, EN→ZH | full 8,000; random half; one expert per domain | chrF | 0.42 — UN expert covers news |

Results are committed in `../results/` as `<task>-screen.json` (and
`samsum-scaling-curve.json`); COMET-22 for the translation runs is in
`mt-comet.json`.

### Shared settings

Every config: LoRA r16 / α32 / dropout 0.05 on q, k, v, o over a frozen base
held in fp32. The configs say `load_in_4bit: true`, but axolotl 0.18 quantises
only for `adapter: qlora`; with `adapter: lora` it loads the base unquantised and
`prepare_model_for_kbit_training` upcasts it to fp32. These are LoRA runs, not
QLoRA (set `ADAPTER=qlora` when submitting `train_screen.sbatch` for real 4-bit,
as the 27B needs). Also:
learning rate 2e-4, cosine with 10 warmup steps; effective batch 16; eager
attention (Gemma 3 under sdpa produced NaNs); `<end_of_turn>` trained as the stop
token; `excess_length_strategy: raise`; `dataset_num_proc: 8`. Three epochs,
except SAMSum (six, so the 100-row arm gets enough steps). `sequence_len` per
task, from the measured token lengths: 1,024 (CLINC150, translation, XSum), 1,536
(SAMSum), 2,048 (CNN/DM), 4,096 (BillSum, BigPatent). Each prepare script checks
every rendered example against `sequence_len` before anything is queued.

### Known problems

- **`dataset_prepared_path: ${MM_PREPARED:-last_run_prepared}` is not expanded by
  axolotl.** Prepared data lands in a directory literally named
  `${MM_PREPARED:-last_run_prepared}` here and in `../self-distill/` (on home
  quota), not on node-local scratch. Axolotl's cache key ignores the base model,
  so 1B and 4B runs on one dataset share it — harmless for Gemma 3 1B and 4B,
  which tokenise identically, but not safe for other model pairs. Not yet fixed.
- **Gemma 3 1B** has 262,145 tokens against 262,144 embedding rows. Axolotl
  resizes the embeddings and the adapter saves them; `../evaluate/hf_peft_provider.py`
  resizes the base to match before loading.
- **Self-distillation with LoRA** has its own problems in axolotl; see
  `../self-distill/README.md`.
