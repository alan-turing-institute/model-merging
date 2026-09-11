# XSum summarisation arm

A second task for the disjoint-halves merging experiment, alongside the Reddit
crime classification arm in the root README.

**The question is the same one:** split a training set in two, fine-tune an
expert on each half in parallel, merge the two experts — do you get back the
model you would have got by training on all the data at once? If you do, then
training is parallelisable across machines in a way that gradient
synchronisation isn't. Call it poor man's parallelism.

**The task is deliberately different.** On crime classification the answer was
one token, so the merge could only fail in one way: predicting the wrong class.
The observed failure was a collapse in recall — the merged model stopped
committing to the positive class, which is what a halved delta looks like when
it pulls the model back toward base behaviour.

Summarisation gives that failure mode somewhere else to show up. The output is
free text, so a merge that has drifted back toward the base model degrades
*stylistically* first: preamble reappears ("Here is a summary of the
article:"), one sentence becomes three, the model starts hedging. XSum's
references are a single sentence, so ROUGE punishes all of that — but the
length and sentence-count statistics that `evaluate_summarisation.py` records
alongside ROUGE will usually move first, and are far easier to interpret than a
ROUGE delta.

## Data

[XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) is 204k BBC News
articles, each paired with the one-sentence summary the BBC itself wrote as
the article's opening line. `prepare_xsum_data.py` subsamples it, renders the
prompts, and splits it exactly the way `prepare_data.py` splits the crime data:

```
../datasets/xsum_dataset     train / validation / test   (the full arm)
../datasets/xsum_dataset1    train / validation          (half 1)
../datasets/xsum_dataset2    train / validation          (half 2)
```

The halves are disjoint and their union is the full set, so "half 1 + half 2 =
full" holds by construction. Sizes are env vars — the defaults (8000 train /
500 validation / 1000 test) are chosen to keep a full cycle to hours rather
than days, not because they're the right sizes for a publishable result:

```bash
XSUM_TRAIN_N=8000 XSUM_VAL_N=500 XSUM_TEST_N=1000 XSUM_MAX_DOC_WORDS=512 XSUM_SEED=42
```

Two details worth knowing before changing anything:

- **Articles are truncated to 512 words.** `sequence_len: 2048` in the training
  configs is a hard cap and axolotl silently drops whatever overflows it, so
  raising `XSUM_MAX_DOC_WORDS` without raising `sequence_len` would start
  truncating away the training target itself. XSum's summary is the article's
  opening sentence, so a cut at 512 words is much less lossy here than it would
  be on a dataset with a genuinely dispersed summary.
- **The prompt is rendered once, in the dataset, as a `prompt` column.** The
  crime arm keeps the prompt wording in two places (`prepare_data.py` and
  `evaluate.py`) and relies on them being edited in lockstep;
  `evaluate_summarisation.py` reads the stored column instead, so training and
  evaluation cannot drift apart.

## Running it

**On a GPU compute instance**, not a laptop. Training is 4-bit
(bitsandbytes) and the merge runs with `--cuda`; neither has a CPU or MPS path,
and preflight will refuse to start without a visible CUDA device. Set
`ALLOW_NO_GPU=1` to run data prep alone somewhere else.

Azure ML compute instances are not reachable by plain `ssh` — the name is not a
DNS hostname, and the az CLI tunnels the connection over a websocket. They are
also normally left stopped, so starting one is an explicit (and billable) step:

```bash
az ml compute start --name mpietrzykA100          # a few minutes; billing starts here
az ml compute connect-ssh --name mpietrzykA100    # add -f <key.pem> if prompted
cd model-merging && screen -S xsum ./run_xsum_pipeline.sh
```

Resource group and workspace are omitted above because `az configure --defaults
group=tire-1 workspace=tire-2` covers them. **Stop the instance when the run
finishes** — an A100 instance bills for wall-clock uptime, not GPU use:

```bash
az ml compute stop --name mpietrzykA100
```

(`screen -S xsum -- ./run_xsum_pipeline.sh` does not work — screen takes the
command directly, and treats the `--` as the program to run.)

Resumable in the same way as `run_pipeline.sh`: every stage skips if its output
is already registered, so a dropped SSH session doesn't cost the whole run.

Smoke-test the whole path first — it exercises every stage in a few minutes and
catches config errors before hours of training:

```bash
XSUM_TRAIN_N=200 XSUM_VAL_N=20 XSUM_TEST_N=20 EVAL_LIMIT=20 KEEP_LOCAL=1 ./run_xsum_pipeline.sh
```

Note that the smoke test registers real (tiny) models under the real names. Bump
the dataset sizes and delete `.pipeline_versions_xsum` to start a proper run —
registration then moves to the next free version rather than overwriting.

Other knobs:

```bash
MERGE_METHODS="linear task_arithmetic ties" ./run_xsum_pipeline.sh   # default: linear
EVAL_BATCH_SIZE=4 ./run_xsum_pipeline.sh                            # lower on CUDA OOM
RESULTS_DIR=/some/path ./run_xsum_pipeline.sh
KEEP_LOCAL=1 ./run_xsum_pipeline.sh                                 # skip the scratch cleanup
```

### What it produces

Registered in the Azure ML workspace:

| Name | What it is |
|---|---|
| `gemma3-xsum-full-lora` | LoRA trained on all the data — the target |
| `gemma3-xsum-1-of-2-lora` | LoRA trained on half 1 — a merge input |
| `gemma3-xsum-2-of-2-lora` | LoRA trained on half 2 — a merge input |
| `gemma3-xsum-full` | the full-data adapter materialised into full weights |
| `gemma3-xsum-merged-<method>` | the two halves merged — the result |

One results JSON per variant in `$RESULTS_DIR`, including the untouched base
model evaluated zero-shot. Six evaluations by default, and all six are needed:
the base model is the floor, the full-data model is the target, and **the two
halves are what the merge has to beat**. Comparing a merge only against the
full-data model can't distinguish "merging works" from "merging landed halfway
between its inputs, and one input was already decent" — which is exactly what
happened on the classification arm.

The half-data adapters are not converted into full models as a separate step —
the merge configs use mergekit's `<base>+<adapter>` syntax. That saves the
conversion step but **not the disk**: `--lora-merge-cache` is where mergekit
materialises each combination, as a full model each, so the bytes are the same
either way.

### Disk

This is the binding constraint, and the first real run died on it. Budget:

| | |
|---|---|
| merge cache | ~17GB — two full materialised models at ~8.1GB each |
| each merge output | ~8.1GB |
| base model in `~/.cache/huggingface` | ~9GB |

The cache is identical across merge methods, so it is built once and reused by
every entry in `MERGE_METHODS`. Two things keep the peak survivable: the
full-data model's local copy is deleted the moment it is registered (it is
durable, and evaluation references it as `azureml:<name>:<version>`), and the
cache can live on a different filesystem:

```bash
MERGE_CACHE_DIR=/tmp/lora-merge-cache ./run_xsum_pipeline.sh
```

On an Azure ML compute instance the root disk is often the tighter of the two —
`df -h / /mnt` before starting. Use `/tmp` rather than `/mnt` directly: it is on
the same filesystem as `/mnt` but is writable by `azureuser`, whereas `/mnt`
itself is root-owned. Note it is a *temp* disk — the cache does not survive an
instance stop/start, and gets rebuilt (~17GB) on the next merge. The pipeline now checks free space before
merging and fails immediately rather than after the training and uploads.

### Timing

Not yet measured. Time the smoke test and scale from it before committing to a
long run: training is roughly linear in `XSUM_TRAIN_N` (the halves are half the
steps of the full arm each, so all three adapters together cost about twice the
full arm alone), and evaluation is linear in `XSUM_TEST_N` times the number of
variants.

## The turn terminator must be trained

Gemma ends a turn with `<end_of_turn>`, not `<eos>`. Axolotl looks for the
model's EOS token in the chat template, doesn't find it, warns that *"the turn
terminator won't be trained"*, and continues. The result is a model that never
learns to stop.

The first full-scale run showed exactly that. The fine-tuned models produced
**51 words and 2.3 sentences** against a 21-word, one-sentence reference —
running to the 64-token generation cap on most examples — and the full-data
model scored **worse than the untouched base** (0.262 vs 0.281 ROUGE-1). A
fine-tune that loses to no fine-tuning answers no question about merging, so
that run is not interpretable as a merging result.

The configs now set:

```yaml
eot_tokens:
  - "<end_of_turn>"
train_on_eos: turn
```

They also use `chat_template: tokenizer_default` rather than the `gemma` alias,
so training formats text exactly as `run_inspect_xsum.py` does at evaluation —
the two rendering paths must agree or the model is scored on text shaped
differently from what it saw.

**Read the length columns before the scores.** `mean_pred_words` near the
reference's ~21 and `mean_pred_sentences` near 1.0 is what a healthy run looks
like; 50 words and 2.5 sentences means the model isn't terminating, whatever
its ROUGE says.

## The evaluator

Evaluation runs through [Inspect AI](https://inspect.aisi.org.uk/) by default —
`evaluate/xsum_task.py`, driven by `evaluate/run_inspect_xsum.py`. It writes a
browsable `.eval` log into `$RESULTS_DIR/inspect-logs` alongside the results
JSON, and reports standard error on ROUGE as well as the mean.

```bash
inspect view --log-dir "$RESULTS_DIR/inspect-logs"
```

`XSUM_EVAL=legacy` switches back to the hand-rolled
`evaluate/evaluate_summarisation.py`. Both write the same results JSON, so
everything downstream is unaffected by the choice. See
[evaluate/README.md](evaluate/README.md) for the details, including why
adapters need the custom `hf-peft` provider.

## Reading the results

```bash
uv run --project evaluate python experiments/bootstrap_rouge.py "$RESULTS_DIR"
```

This is the summarisation counterpart to `experiments/mcnemar.py`. McNemar's
test doesn't apply — ROUGE is a continuous per-example score, not a right/wrong
bit — but the pairing is just as real, since every model is scored on the same
articles. `bootstrap_rouge.py` does a paired bootstrap over the per-example
score differences and reports a confidence interval on the difference, not just
a p-value.

The table it prints also carries `words` and `sents` per prediction. Read those
first: if the merged model's mean sentence count has climbed toward the base
model's while the fine-tuned models sit near 1.0, the merge has drifted back
toward base behaviour, and that is a more legible statement of the problem than
any ROUGE delta.

Then actually read some generations — they're in the `predictions` field of
every results JSON, and `evaluate_summarisation.py` prints a few at the end of
each run (`--show N`). A ROUGE gap of a few tenths of a point is not
necessarily a difference a reader would notice.

**The caveat from the classification arm applies here in full:** these are n=1
per condition with no seed variation. On crime classification the two halves
differed from each other by more than the effect being measured, which made the
headline comparison uninterpretable. Expect to need
`experiments/seed_sweep.sh`-style repeats here too before treating any merge
result as established, and check the two halves against each other *first* —
if they differ a lot, nothing downstream of them means much yet.

## Files

| Path | |
|---|---|
| `prepare_xsum_data.py` | download, subsample, render prompts, split into halves |
| `train/xsum_gemma.yaml` | full-data QLoRA config |
| `train/xsum_gemma{1,2}.yaml` | the two half-data configs (dataset paths + `output_dir` differ, nothing else) |
| `merge/merge_linear_xsum_config.yaml` | the primary merge |
| `merge/merge_task_arithmetic_xsum_config.yaml` | deltas rather than raw weights |
| `merge/merge_task_arithmetic_w1_xsum_config.yaml` | both weights at 1.0 — a direct test of the halved-delta hypothesis |
| `merge/merge_ties_xsum_config.yaml` | trim + sign-resolve before summing |
| `evaluate/evaluate_summarisation.py` | ROUGE + length statistics, one JSON per variant |
| `experiments/bootstrap_rouge.py` | paired bootstrap between result JSONs |
| `run_xsum_pipeline.sh` | the whole thing, resumable |
| `pipeline_lib.sh` | shared pipeline helpers (see note below) |

`pipeline_lib.sh` holds the results-directory resolution, preflight checks and
Azure ML version bookkeeping that `run_xsum_pipeline.sh` uses.
`run_pipeline.sh` predates it and still carries its own copy of the same
helpers; switching it over is a safe follow-up, but it was left alone here
rather than edited mid-experiment.

## Differences from the crime configs, and why

The training configs are otherwise identical to `crime_gemma*.yaml`:

- **`gradient_checkpointing: true`** — the crime configs omit it, which is why
  they OOM on a T4's 16GB at this batch size and sequence length. Costs ~20%
  throughput on an A100 and makes the config portable to the smaller box.
- **`train_on_inputs: false`** — stated explicitly. The `chat_template`
  strategy already defaults to training on the assistant turn only, but the
  cost of getting it wrong is much higher here than on classification: the
  input is a BBC article, so training on it would spend most of the gradient
  on reproducing news copy rather than on summarising.
- **`num_epochs: 2` rather than 3** — the target is one sentence, so there is
  much less to fit, and a third pass mostly buys memorisation of BBC house
  style. Worth raising if the halves look under-trained relative to full.
