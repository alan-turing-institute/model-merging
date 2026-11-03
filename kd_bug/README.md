# Off-by-one in Axolotl's KD plugin, reproduced on its own example

Axolotl's knowledge-distillation plugin (`axolotl.integrations.kd`, checked on
`axolotl==0.18.0`) stores each teacher distribution one position too late. The
student's prediction at index `t` (which predicts `input_ids[t+1]`) is trained
to match the teacher's distribution for `input_ids[t]`, the token it has just
read. Models trained this way learn to repeat themselves (`the the`,
`family family family`).

This directory reproduces the bug using only the plugin README's configuration
and its example dataset,
[`axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample`](https://huggingface.co/datasets/axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample).
The dataset is not modified.

## Run it

```bash
uv sync
sbatch run.sbatch          # preprocess, then both checks; ~5 min on one GH200
```

Or step by step:

```bash
PYTHONPATH=. uv run axolotl preprocess kd_readme_example.yaml   # CPU
uv run python inspect_prepared.py                               # CPU, no model
uv run python score_alignment.py                                # GPU
```

`meta-llama/Llama-3.2-1B-Instruct` stands in for the teacher's
tokenizer: it shares the Llama-3 vocabulary with the teacher that generated the
dataset (`Llama-3.3-70B-Instruct`), and it is small enough to run.

## Where the bug is

`ChatTemplateStrategyWithKD.transform_logprobs` (and the identical v2 version)
in `integrations/kd/chat_template.py` places the teacher rows using lengths
alone:

```python
input_padding_len = input_seq_len - target_seq_len
```

The example dataset has one row per generated token, so row `j` (the
distribution that produced assistant token `j`) lands at index `P + j`, which is
the index of that same token.

The loss in `integrations/kd/kernels/liger.py` then pairs
`target_logprobs[t]` with the student hidden state at `t` without shifting.
The cross-entropy labels in the same function *are* shifted:

```python
true_labels = torch.nn.functional.pad(true_labels, (0, 1), value=ignore_index)
true_labels_flat = true_labels[:, 1:].contiguous().view(-1)
```

So the CE term is aligned, and the KD term is one position late.

## Evidence

### 1. The prepared tensors (`inspect_prepared.py`, no model)

The teacher generated these responses, so its top-1 at each step is usually the
token it emitted. That shows which token each prepared row is actually about.
Across all 742 examples:

```
unmasked KD positions t: 621951
  teacher top-1 == input_ids[t]    (the token already consumed): 574524  ( 92.4%)
  teacher top-1 == input_ids[t+1]  (the token the loss targets):    2042  (  0.3%)
```

Position by position, the loss asks the student to predict the next token
(column 3) while matching a teacher distribution whose top-1 is the current
token (column 4):

```
      t      input_ids[t]  student predicts   teacher row top-1
    137              'To'        ' address'               'The'
    138        ' address'            ' the'          ' address'
    139            ' the'          ' issue'              ' the'
    140          ' issue'             ' of'            ' issue'
    141             ' of'            ' the'               ' of'
```

### 2. Against a real model (`score_alignment.py`, GPU)

This scores teacher row `t` against Llama-3.2-1B-Instruct's prediction at index
`t + shift`, over 32 examples:

```
  shift  mean KL (nats)  top-1 agreement
     -2          7.0662             0.8%
     -1          0.4314            82.1%  <- where the rows belong
     +0         12.1427             0.3%  <- what the KD loss trains on
     +1          9.8581             0.7%
```

## Also: the README example config does not load

The plugin README gives `type: "axolotl.integrations.kd.chat_template"`. That
resolves to the v2 loader, which expects a separate `target_token_ids` column,
so on the README's own example dataset it fails with `KeyError: 'target_token_ids'`.
The dataset is in the legacy `[{"token": "token_id:###", "logprob": ...}]`
format, and `load_legacy` cannot be selected from `type:`, because the resolver
only honours a `.load_*` suffix for modules inside `axolotl.prompt_strategies`.
`kd_shim/legacy.py` re-exports `load_legacy` in one line so it can be reached.
Both loaders place the rows with identical code, so this does not affect the
off-by-one.

## Workaround and fix

**Workaround on the dataset side:** emit one extra logprob row per example, so
that `len(logprobs) == len(full) - len(prompt) + 1`. This moves every row one
index earlier. The extra row lands on the final position, which is masked
(`labels[-1] == -100`). See `../self-distill/KD_ALIGNMENT_BUG.md`.

**Upstream fix:** shift the KD targets the same way the CE labels are already
shifted. Either place row `j` at `P + j - 1` in `transform_logprobs`, or pair
`target_*[t+1]` with the hidden state at `t` in the loss.

## Files

| file | purpose |
| --- | --- |
| `kd_readme_example.yaml` | the plugin README config, on its example dataset |
| `kd_shim/legacy.py` | one-line re-export so `type:` can reach `load_legacy` |
| `inspect_prepared.py` | model-free check on the tensors Axolotl trains on |
| `score_alignment.py` | KL of the prepared rows against a real Llama-3 model |
| `run.sbatch` | runs all three steps on one GPU |
