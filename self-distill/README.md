# Self-distillation on XSum

We perform self-distillation using the kd plugin in Axolotl to train Gemma 3 to perform one sentence summarisation of BBC News articles using (some of) the XSum dataset.

The aim is to get an output which is

- a one sentence summarisation
- between 20-25 words
- no preamble, i.e. no "My one sentence summarisation is: ..."

NB this uses
This was split out of [`../train`](../train), which remains the crime-data
fine-tuning project. The two have separate uv environments; this one also needs
`vllm` (for teacher generation) and `torchao` (see below).

## Pipeline

#### Data preparation

We first need to prepare the data.  The XSum dataset is 200k rows.  We take a 10k sample of this for our use and generate suitable prompts.  We output the teacher prompts together with a short student prompt and a longer student prompt (the long version is just to check the self-distillation is working correctly).

```bash
uv run python prepare_data_XSum.py EdinburghNLP/xsum ../datasets/xsum_prompts  # build prompt variants
```

The first argument is the dataset to load, either a Hugging Face address or a local folder saved with `save_to_disk`; the second is where to save the prompts. The sample sizes can be changed with `--train-size`, `--validation-size` and `--test-size` (defaults 10000, 1000 and 1000).

#### Splitting the data (for merging)

To train several students on disjoint parts of the data and then merge them, split the prompts dataset into `n` pieces:

```bash
uv run python split_dataset.py n <input_dir> <output_dir>
```

which with split your dataset into `n` peices and saves them in the output directory as `<dataset_name>_i_of_n`.  It will handle `DatasetDicts`, but deliberately drops the `test` paertition and only splits the `train` and `validation` parts.  Pieces are contiguous blocks in the existing order (the prompts are already shuffled); pass `--seed` to shuffle first.

#### Teacher training

First you must ensure that the CUDA driver is properly pointing to the latest version (see below for details).

```bash
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat
```

For self-distillation, we need a teacher and a student.  First, we train the teacher with a strict prompt and a system prompt detailing the summarisation task.

```bash
uv run python generate_teacher_logprobs.py google/gemma-3-4b-it \
  $PROJECTDIR/$USER/model-merging/datasets/xsum_prompts \
  $PROJECTDIR/$USER/model-merging/datasets \
  --prefix xsum_kd_data
```

The arguments are the teacher model (a Hugging Face address or a local folder), the prompts dataset and the output directory. One dataset is written per student prompt, named `<prefix>_long` and `<prefix>_short`; with `--prefix xsum_kd_data` and `../datasets` as above, these are the paths the Axolotl configs read from. For a split dataset, run it once per piece, e.g.

```bash
uv run python generate_teacher_logprobs.py google/gemma-3-4b-it \
  $PROJECTDIR/$USER/model-merging/datasets/xsum_prompts_1_of_3 \
  $PROJECTDIR/$USER/model-merging/datasets \
  --prefix xsum_kd_data_1_of_3
```

Optional arguments (see `--help`): `--top-k` (default 20), `--temperature` (0, i.e. greedy), `--max-tokens` (128), `--max-seq-len` (4096, which must match `sequence_len` in the Axolotl config), `--seed`, `--prefix`, `--teacher-field` and `--student-fields` (for other column names), `--tensor-parallel-size`, `--gpu-memory-utilization` and `--no-enforce-eager`.

This needs a GPU and the forward-compatibility driver on `LD_LIBRARY_PATH` (see [Isambard technicalities](#isambard-technicalities)). Note that vLLM is not exactly deterministic between runs, even when greedy: rerunning can give a slightly different generation for a few rows.

We record the one sentence summarisation together with the `logprobs`.  This is a sequence of probability distributions, one for each token in the output.

- `tok_k=20` takes the highest 20 probabilities in the probability distribution only.

#### Student training

The student is trained on both the `longprobs` and the output of the teacher.

NB On Isambard this should be submitted as a job, otherwise it will run out of memory and cause your session to be killed.  We provide a short bash script for this.

```bash
sbatch train_xsum.sbatch self-distillation_xsum.yaml \
  $PROJECTDIR/$USER/model-merging/models/gemma3_xsum_short \
  $PROJECTDIR/$USER/model-merging/datasets/xsum_kd_data_short
```

This takes as parameters:

- a yaml config file for the Axolotl training using the kd plugin
- an output directory for the model
- optionally a dataset directory.  If the dataset directory is given, the script writes a new yaml idenitcal to the one given, except for the dataset location.

### Isambard technicalities

Currently on Isambard, CUDA version 12.7 is used (driver 565.57.01).  However, Axolotl requires CUDA version 13, so we have used a [forward-compatibility driver](https://docs.isambard.ac.uk/user-documentation/guides/gpus_and_cuda/#cuda-forward-compatibility).  This is installed in `$LIBRARYDIR/shared` and is accessable to everyone in TIRE on Isambard.  To enable it you must set `LD_LIBRARY_PATH` in **every** shell and **every** job script (check this):

```bash
CUDA13=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0
export LD_LIBRARY_PATH=$CUDA13/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}     # This is what is required
export CUDA_HOME=$CUDA13                                                       # This adds the CUDA toolkit and may be useful later
export PATH=$CUDA13/bin:$PATH
```

If you don't do this, axolotl will revert to CPUs and hence take a long time.  Elsewhere you may get odd errors which you might well attribute to something else.

### The alignment problem in the kd plugin

The KD plugin never receives an explicit offset. It infers where the teacher
distributions belong from lengths alone, so every row must satisfy exactly:

```
len(teacher_logprobs) == len(template(messages_combined))
                         - len(template(student_messages, add_generation_prompt=True))
```

Gemma's chat template emits a `\n` after `<end_of_turn>`, so the templated
assistant turn is one token longer than the generation. `generate_teacher_logprobs.py`
closes that gap with a one-hot row per leftover template token and asserts the
identity per row. Without it every teacher distribution lands on the wrong token
and training silently learns noise.

We have now fixed this bug by adapting for it in the `generate_teacher_logprobs.py` script.

### Teacher and student see different prompts

`prepare_data_XSum.py` emits three prompt variants per article. The teacher is
conditioned on a strict system prompt; the student is trained on a plainer one
and has to learn the format from its weights instead of reading it each time.
A longer student prompt is included as a check on whether the self-distillation is operating correctly.
For details see the `prepare_data_XSum.py` file.

Different variations on the teacher propmt were tried.  This one produced the most clean outputs: no preamble, no markdown, one sentence, 20-25 words.

## Self-distilled LoRA (`kd_screen.sbatch`)

A self-distilled LoRA built to match `../train/xsum_screen_gemma.yaml`'s
reference-trained LoRA in everything except the training signal: the same 4,000
XSum articles cut to 512 words, the same adapter, schedule and batch, scored on
the same 1,000 test articles. The teacher, prompts and loss are main's:
`generate_teacher_logprobs.py` unchanged, 0.5 KL + 0.5 cross-entropy,
temperature 1, top-20.

### Running it

```bash
cd self-distill
uv run --project ../train --no-sync python prepare_kd_screen_prompts.py \
    ../datasets/xsum_screen/full ../datasets/xsum_kd_screen_prompts          # login node
sbatch --job-name=kd-teacher kd_screen.sbatch teacher                       # teacher targets and top-20 logprobs
sbatch --job-name=kd-train   kd_screen.sbatch trainloop                     # bf16 LoRA, own loop (recommended)
sbatch --job-name=kd-train   kd_screen.sbatch trainloop google/gemma-3-1b-it ../models/gemma3-xsum_kd_screen_1b_bf16loop-lora
sbatch --job-name=kd-train   kd_screen.sbatch train self-distillation_xsum_lora.yaml   # axolotl, 4-bit
cd ../evaluate && sbatch evaluate.sbatch xsum google/gemma-3-4b-it \
    --adapter ../models/gemma3-xsum_kd_screen_bf16loop-lora \
    -T prompt_variant=short -T dataset_path=../datasets/xsum_screen/test
uv run python xsum_kd_compare.py logs
```

The job uses the self-distill environment at
`$PROJECTDIR/$USER/mm/venvs/self-distill` (project storage, not home). Build it
once with
`UV_PROJECT_ENVIRONMENT=$PROJECTDIR/$USER/mm/venvs/self-distill uv sync --frozen`.
Its teacher stage passes `--max-seq-len 1024`, which must equal the training
`sequence_len`: anything truncated has its teacher logprobs misplaced.

### Train with `kd_lora_train.py`, not axolotl

Axolotl 0.18's KD trainer is unreliable with a LoRA on Gemma 3:

| Configuration | Outcome |
|---|---|
| bf16 base (`self-distillation_xsum_lora_bf16.yaml`) | NaN gradient at step 1, in bf16 and fp32, with or without padding or gradient checkpointing |
| 8-bit base | trained, but validation loss rose from 1.32 to 2.92 and the model collapsed (3.7-word outputs) |
| 4-bit base | trained; raw gradient norms in the hundreds, 0.8% of outputs loop |

The same step built by hand is finite (`kd_lora_nan_check.py`), so
`kd_lora_train.py` trains the configuration outside axolotl. It reuses axolotl's
prepared tensors, the same fused loss and the same optimiser and schedule. Two
deliberate departures: the loss gets fp32 hidden states and an fp32 copy of the
output layer, as axolotl's working 4-bit path effectively does; and a step with a
non-finite gradient is skipped and logged, with an abort after ten. On XSum it
trained smoothly (validation loss 5.00 → 0.175, 5 of 747 steps skipped) and
produced no looping outputs. For a smaller student on the 4B teacher's data, it
checks the largest teacher token id against the embedding and resizes it if
needed.

Axolotl's logged KD losses run about 1.44× higher than the per-token loss of the
same rows, because of how it averages across batches. Compare runs by their test
scores, not their logged losses.

### Checks

- **`kd_step0_check.py`** (`kd_screen.sbatch check`): with the untrained student,
  the per-token KL at the data's alignment and one position either side; the
  trainer's logit path against the model's; cross-entropy; axolotl's fused loss;
  and the same rows scored behind the teacher's own prompt. Measured: alignment
  correct (KL 0.003 nats in the teacher's context; 15–28 at ±1), logged loss
  exactly 0.5 KL + 0.5 CE. Quantisation alone puts the student 0.066 (8-bit) and
  0.50 (4-bit) nats from the teacher.
- **`kd_grad_check.py`**, **`kd_lora_nan_check.py`**, and
  `kd_screen.sbatch probe` located the axolotl NaN described above.

The large step-0 loss is not a bug. The teacher writes under a strict system
prompt, and the student reads the short one; closing that gap is what this
training does. `self-distillation_xsum.yaml`'s comment that the step-0 KD loss
should be near zero holds only when the student is given the teacher's prompt.

### Results

`../results/xsum-kd-compare.txt`. On 1,000 test articles, the bf16 self-distilled
LoRA reaches ROUGE-1 0.241 (the teacher's own targets score 0.235 against the
references) against 0.400 for the reference-trained LoRA. Its entity support is
0.859 against 0.686: it invents far fewer names and numbers.
