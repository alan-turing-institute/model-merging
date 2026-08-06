# MergeKit Handover Notebook

A self-contained, runnable walkthrough for merging language models with [MergeKit](https://github.com/arcee-ai/mergekit). Written so a developer who has already `pip install`-ed MergeKit can pick this up, run it against their own local models, and add a new model of their own — without needing anyone else's credentials.

Each section below is a "cell": prose explaining the *why*, followed by a code block you can paste directly into a shell or notebook.

> **Security note baked into this document:** no API keys, tokens, or secrets appear anywhere below. Every place a credential is needed uses a placeholder and an environment-variable pattern, so nothing sensitive is ever pasted into a config file, a notebook cell, or version control.

---

## 1. What MergeKit actually does

MergeKit combines the *weights* of two or more pretrained models into a single new model — no training loop, no gradient updates, no labeled dataset required for the standard merge methods. It reads each source model's tensors, combines them according to a chosen algorithm (linear averaging, spherical interpolation, task-arithmetic, TIES, DARE-TIES, passthrough/"frankenmerge" layer-stacking, etc.), and writes out a new model directory in standard Hugging Face format.

The only "data" most merges need is: the model weights themselves (local disk or downloaded from the Hugging Face Hub) and a YAML config describing how to combine them. (Section 7 covers the one path where an actual evaluation dataset *is* used — evolutionary merge search.)

---

## 2. Verify the install

You said MergeKit is already installed — this just confirms the CLI is on `PATH` and shows the version, so later steps don't fail on a missing/broken install.

```bash
# Confirm the CLI entry points exist
mergekit-yaml --help
mergekit-evolve --help   # optional, only needed for Section 7

# Confirm the package + its pinned dependencies (torch, transformers, etc.)
python -c "import mergekit; print(mergekit.__file__)"
pip show mergekit
```

If any of these fail, the install itself is broken — that's a separate fix (`pip install -U mergekit`, or reinstall from source with `pip install -e .` inside a fresh clone of the repo) and out of scope for this handover.

---

## 3. Anatomy of a merge config

Every merge is driven by one YAML file. The four things that matter:

- `merge_method` — the algorithm (`linear`, `slerp`, `task_arithmetic`, `ties`, `dare_ties`, `passthrough`, `model_breadcrumbs`, …)
- `models` — the list of source models, each either a Hugging Face Hub repo ID (`org/model-name`) or a local directory path
- `base_model` — required by some methods (`task_arithmetic`, `ties`, `dare_ties`) as the reference point the others are diffed against
- `parameters` — per-model or global weights/knobs (e.g. `weight`, `density`)
- `tokenizer_source` — which tokenizer/vocabulary to standardize on (see the warning right below — skipping this is the single most common way a merge silently breaks)

```yaml
# config.yaml — a minimal linear merge of two public models
merge_method: linear
models:
  - model: mistralai/Mistral-7B-v0.1
    parameters:
      weight: 0.5
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      weight: 0.5
tokenizer_source: union
dtype: float16
```

> **Always check `vocab_size` matches before merging, or set `tokenizer_source` explicitly.** These two specific models are a real example of the mismatch: `mistralai/Mistral-7B-v0.1` has `vocab_size: 32000`, but `teknium/OpenHermes-2.5-Mistral-7B` adds two ChatML special tokens and reports `vocab_size: 32002`. Without `tokenizer_source`, MergeKit copies one model's tokenizer into the output while the merged `embed_tokens`/`lm_head` weights keep a different model's shape — the merge succeeds with no warning, and the mismatch only surfaces later, when `transformers` refuses to load it. `tokenizer_source: union` builds a combined vocabulary and resizes every source model's embedding matrix to match before merging, so nothing gets silently dropped or misaligned. Use `tokenizer_source: base` instead if you'd rather standardize on one model's vocab and don't need the other model's extra tokens.

Swap in your own two models (Hub IDs or local paths) to run this against whatever you actually have on hand — that's "the available data" from MergeKit's point of view: the model checkpoints, not a labeled dataset.

---

## 4. Running the merge

```bash
mergekit-yaml config.yaml ./merged-model \
  --cuda \
  --lazy-unpickle \
  --copy-tokenizer \
  --out-shard-size 1B
```

What each flag buys you:

| Flag | Why you'd want it |
|---|---|
| `--cuda` | Do the tensor math on GPU instead of CPU — much faster if one is available |
| `--lazy-unpickle` | Stream weights from disk instead of loading everything into RAM at once — use this if you hit OOM on a large model pair |
| `--low-cpu-memory` | Further reduces peak RAM at some speed cost; combine with `--lazy-unpickle` for the tightest memory budget |
| `--copy-tokenizer` | Copies the tokenizer from the base model into the output directory, so the merged model is immediately loadable |
| `--out-shard-size` | Controls output file chunking — useful for very large models or slow storage |
| `--allow-crimes` | Required for some merges across models with mismatched vocab/architecture — only use if you understand why the models differ |

The output directory (`./merged-model` above) ends up looking like any normal Hugging Face model folder — `config.json`, `*.safetensors`, tokenizer files.

---

## 5. Smoke-testing the merged model

Before running any real evaluation, load it and generate a few tokens — this catches broken merges (garbage output, load errors) in seconds rather than after a long eval run.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./merged-model"

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "In one sentence, explain what model merging does:"
inputs = tok(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tok.decode(output[0], skip_special_tokens=True))
```

If this prints coherent text, the merge produced a structurally valid model. Coherent-but-bad output is a *merge quality* problem (wrong method, bad weights) rather than a *broken pipeline* problem — that's a separate tuning question, not covered here.

---

## 6. Running against your own local checkpoints

If the models you want to merge already live on disk (rather than being pulled from the Hub), just point `model:` at the local directory instead of a Hub ID — MergeKit treats both identically:

```yaml
# config.yaml — merging two local checkpoints
merge_method: ties
base_model: /path/to/your/base-model
models:
  - model: /path/to/your/base-model
    parameters:
      weight: 1.0
  - model: /path/to/your/finetuned-checkpoint
    parameters:
      weight: 1.0
      density: 0.5
parameters:
  normalize: true
tokenizer_source: base
dtype: float16
```

A finetune of the same base model usually keeps the same vocab size, so this is lower-risk than Section 3's cross-family example — but setting `tokenizer_source` explicitly costs nothing and removes the guesswork.

Run it exactly the same way as Section 4:

```bash
mergekit-yaml config.yaml ./merged-model-local --cuda --lazy-unpickle --copy-tokenizer
```

`ties` and `dare_ties` both need a `base_model` because they work by measuring how far each model *deviates* from that base, then merging the deviations — that's why the config above looks different from the plain `linear` example.

---

## 7. Optional: scoring merges against an actual dataset (evolutionary merge)

If you want MergeKit to *search* for good merge parameters rather than you hand-picking weights, `mergekit-evolve` runs an evolutionary search that scores each candidate merge against a real evaluation task (via [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)). This is the one place actual benchmark data enters the picture.

```yaml
# evolve.yaml
genome:
  models:
    - mistralai/Mistral-7B-v0.1
    - teknium/OpenHermes-2.5-Mistral-7B
  merge_method: task_arithmetic
  base_model: mistralai/Mistral-7B-v0.1
tasks:
  - name: gsm8k
    weight: 1.0
```

```bash
mergekit-evolve evolve.yaml \
  --storage-path ./evolve-workdir \
  --num-fewshot 5 \
  --generations 20
```

This is slower and more resource-intensive than a plain merge (it runs many candidate merges and evaluates each one), so only reach for it once the basic pipeline in Sections 3–5 is working end-to-end.

---

## 8. Adding an additional model

### 8a. Public model — no credentials needed

Just add another entry to `models:`. MergeKit downloads it from the Hub automatically the first time it's referenced:

```yaml
merge_method: linear
models:
  - model: mistralai/Mistral-7B-v0.1
    parameters:
      weight: 0.34
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      weight: 0.33
  - model: NousResearch/Nous-Hermes-2-Mistral-7B-DPO   # <- the new model
    parameters:
      weight: 0.33
dtype: float16
```

For `linear`, weights are typically normalized to sum to 1 across all models — adjust the others when you add a new one.

### 8b. Gated or private model — bring your own token

Some models on the Hub require you to accept a license and authenticate before downloading. **Do this with your own token, stored outside the codebase — never inside a YAML config or a notebook cell.**

The recommended path is the Hub CLI's own secure login, which stores the token in your local Hugging Face cache (`~/.cache/huggingface/token`), not in this project:

```bash
huggingface-cli login
# You'll be prompted interactively to paste your own token.
# Get one at https://huggingface.co/settings/tokens if you don't have one yet.
```

MergeKit (via `huggingface_hub`) picks this up automatically — no config changes needed once you're logged in.

If you'd rather use an environment variable (e.g. in a CI job or a container where interactive login isn't possible), set it in your own shell/session — **not** in a file that gets committed:

```bash
export HF_TOKEN="hf_REPLACE_WITH_YOUR_OWN_TOKEN"
```

If you prefer a `.env` file for local dev, create one that is never committed:

```bash
# .env  (add this filename to .gitignore — see Section 9)
HF_TOKEN=hf_REPLACE_WITH_YOUR_OWN_TOKEN
```

```python
# load it in Python if you're driving mergekit programmatically
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ, nothing hardcoded
```

Then reference the new gated model exactly like any other entry in `models:` — MergeKit doesn't need the token mentioned in the YAML at all, it just needs to find it in your environment/cache when it tries to download.

### 8c. A model you already have locally

Same as Section 6 — just use the local directory path instead of a Hub ID. No token needed either way.

---

## 9. Hygiene checklist before handing this off further

- [ ] `.env`, `HF_TOKEN`, and anything under `~/.cache/huggingface/` are **not** committed
- [ ] `.gitignore` includes at minimum:
  ```
  .env
  merged-model/
  evolve-workdir/
  *.safetensors
  ```
- [ ] Large output model directories are stored somewhere other than the git repo (object storage, a model registry, or just a local scratch path)
- [ ] Anyone re-running this authenticates with **their own** Hugging Face account, not a shared key

---

## 10. Troubleshooting quick-reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Merge running whole models into GPU memory at once | Add `--lazy-unpickle --low-cpu-memory`, or drop `--cuda` and merge on CPU (slower but far less memory pressure) |
| `KeyError` / shape mismatch on merge | Source models have different architectures or vocab sizes | Only merge models from the same base family unless you know the specific method supports it; `--allow-crimes` exists but is not a general fix |
| Merged model loads but outputs garbage | Bad merge weights/method for this model pair, not a pipeline bug | Try a different `merge_method`, adjust `weight`/`density`, or fall back to Section 7's search instead of guessing |
| `401 Unauthorized` downloading a model | Not authenticated, or haven't accepted the model's license on the Hub | Run `huggingface-cli login` with your own token, and visit the model page on the Hub to accept its license first |
| Tokenizer missing from output | Forgot `--copy-tokenizer` | Re-run with the flag, or manually copy tokenizer files from the base model into the output directory |
| `RuntimeError: ... ignore_mismatched_sizes ...` / `embed_tokens.weight MISMATCH ... ckpt: torch.Size([N, H]) vs model: torch.Size([M, H])` when loading the merged model | Source models have different `vocab_size` and the config didn't set `tokenizer_source`, so the copied tokenizer and the merged embedding weights disagree on vocab size | Don't just load with `ignore_mismatched_sizes=True` — that silently reinitializes the mismatched rows with random weights. Add `tokenizer_source: union` (or `base`) to the config and **re-run the merge**; the broken output already on disk needs to be regenerated, not patched at load time |
