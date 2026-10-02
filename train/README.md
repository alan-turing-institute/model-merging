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
