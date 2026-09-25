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

For self-distillation, we need a teacher and a student.  First, we train the teacher with a strict prompt and a system prompt detailing the summarisation task.

```bash
uv run python generate_teacher_logprobs.py google/gemma-3-1b-it ../datasets/xsum_prompts ../datasets --prefix xsum_kd_data
```

The arguments are the teacher model (a Hugging Face address or a local folder), the prompts dataset and the output directory. One dataset is written per student prompt, named `<prefix>_long` and `<prefix>_short`; with `--prefix xsum_kd_data` and `../datasets` as above, these are the paths the Axolotl configs read from. For a split dataset, run it once per piece, e.g.

```bash
uv run python generate_teacher_logprobs.py google/gemma-3-1b-it \
    ../datasets/xsum_splits/xsum_prompts_1_of_3 ../datasets/xsum_kd_1_of_3
```

Optional arguments (see `--help`): `--top-k` (default 20), `--temperature` (0, i.e. greedy), `--max-tokens` (128), `--max-seq-len` (4096, which must match `sequence_len` in the Axolotl config), `--seed`, `--prefix`, `--teacher-field` and `--student-fields` (for other column names), `--tensor-parallel-size`, `--gpu-memory-utilization` and `--no-enforce-eager`.

This needs a GPU and the forward-compatibility driver on `LD_LIBRARY_PATH` (see [Isambard technicalities](#isambard-technicalities)). Note that vLLM is not exactly deterministic between runs, even when greedy: rerunning can give a slightly different generation for a few rows.

We record the one sentence summarisation together with the `logprobs`.  This is a sequence of probability distributions, one for each token in the output.

 - `tok_k=20` takes the highest 20 probabilities in the probability distribution only.

#### Student training

The student is trained on both the `longprobs` and the output of the teacher.

NB On Isambard this should be submitted as a job, otherwise it will run out of memory and cause your session to be killed.  We provide a short bash script for this.

```bash
sbatch train_xsum.sbatch short                 # or: long
```

The output is written to ../models.

### Isambard technicalities

Currently on Isambard, CUDA version 12.7 is used (driver 565.57.01).  However, Axolotl requires CUDA version 13, so we have used a [forward-compatibility driver](https://docs.isambard.ac.uk/user-documentation/guides/gpus_and_cuda/#cuda-forward-compatibility).  This is installed in `$LIBRARYDIR/shared` and is accessable to everyone in TIRE on Isambard.  To enable it you must set `LD_LIBRARY_PATH` in **every** shell and **every** job script (check this):

```bash
export LD_LIBRARY_PATH=/projects/u6ui/shared/nvhpc/Linux_aarch64/25.11/cuda/13.0/compat
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
