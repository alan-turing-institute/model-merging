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
uv run python prepare_data_XSum.py            # build prompt variants
```

#### Teacher training

For self-distillation, we need a teacher and a student.  First, we train the teacher with a strict prompt and a system prompt detailing the summarisation task.

```bash
uv run python generate_teacher_logprobs_XSum.py
```

We record the one sentence summarisation together with the `longprobs`.  This is a sequence of probability distributions, one for each token in the output.

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
assistant turn is one token longer than the generation. `generate_teacher_logprobs_XSum.py`
closes that gap with a one-hot row per leftover template token and asserts the
identity per row. Without it every teacher distribution lands on the wrong token
and training silently learns noise.

We have now fixed this bug by adapting for it in the `generate_teacher_logprobs_XSum.py` script.

### Teacher and student see different prompts

`prepare_data_XSum.py` emits three prompt variants per article. The teacher is
conditioned on a strict system prompt; the student is trained on a plainer one
and has to learn the format from its weights instead of reading it each time.
A longer student prompt is included as a check on whether the self-distillation is operating correctly.
For details see the `prepare_data_XSum.py` file.

Different variations on the teacher propmt were tried.  This one produced the most clean outputs: no preamble, no markdown, one sentence, 20-25 words.
