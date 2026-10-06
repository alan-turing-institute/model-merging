# `experiments/` — analysis and metric scripts

Two generations of code live here. **Read this before running anything**: the
summarisation metric scripts were written against a pipeline this repo no
longer uses, and they do not read the Inspect `.eval` logs that `evaluate/`
now writes.

---

## Which to use

| I want to… | Use |
|---|---|
| Score summarisation with faithfulness metrics, a judge, or UniEval | **`evaluate/xsum_faithfulness.py`** — an Inspect scorer, run inline with the rest of evaluation |
| Re-analyse crime/sharding results already measured | the recovery scripts below — still current |
| Re-score old Isambard JSON | the legacy scripts below — superseded, kept for provenance |

### The current path

`evaluate/xsum_faithfulness.py` adds BERTScore, NLI entailment, SummaC, a
G-Eval style judge, and UniEval's four dimensions to the existing XSum task.
It is off by default because it loads ~6 GB of extra models:

```bash
cd evaluate
sbatch evaluate.sbatch xsum <model> -T prompt_variant=short -T faithfulness=true
```

Results land in the same `.eval` log as ROUGE. There is no separate re-scoring
pass, no intermediate JSON, and no second table to reconcile.

---

## Files

### Current — sharding and merge analysis

These read the crime results JSON (`metrics.confusion_matrix` and friends) and
are unaffected by the summarisation pipeline change.

| File | What it does |
|---|---|
| `fourway_recovery.py` | Four-way crime sharding: merge minus **best shard**, paired by expert seed, with the recovery fraction *R*. Also derives sensitivity and specificity from each stored confusion matrix. |
| `twoway_recovery.py` | The same comparison at a two-way split. |
| `prereg_twoway.py` | The pre-registered replication: fixed *n*, fixed decision rule, applied to fresh seeds. |
| `mcnemar.py` | Exact McNemar test per method, for the benchmark comparisons. |
| `bootstrap_rouge.py` | Paired bootstrap over article resamples. |
| `inference_cost.py` | Parameters, peak memory and throughput per serving configuration, interleaved across repeats. |
| `adapter_routing.py` | Oracle-router ceiling: each expert scored on its own task. |
| `crosstask_table.py` | Cross-task retention, raw and as a share of what fine-tuning added. |
| `degeneracy.py` | Empty, repetitive and unterminated output counters. |
| `compare_outputs.py` | Side-by-side diff of two arms' generations. |
| `loss_magnitudes.py`, `kd_alignment_smoke.py` | Distillation diagnostics: term magnitudes, and the teacher-equals-student alignment gate. |
| `merge_methods.sh`, `seed_sweep.sh` | Drivers that sweep methods and seeds. |

### Legacy — superseded by `evaluate/xsum_faithfulness.py`

These re-score **stored generations** from the retired Isambard pipeline. They
expect one JSON per arm containing `predictions`, `references` and
`metrics.rouge1`. **No file in this repo is in that format**: the committed
`results/isambard/*.json` carry `metrics` only, so these cannot be run from a
clean checkout without supplying outputs from elsewhere.

| File | What it does | Why superseded |
|---|---|---|
| `metric_bakeoff.py` | Perturbs a reference summary in ways whose ground truth is known — wrong name, wrong number, negation, shuffled word order, unrelated summary — and scores nine metrics on identical inputs. The source of the finding that ROUGE-1 is the only metric that does not move at all under shuffling. | Still useful as a *metric validation* tool; it does not score models, so it has no Inspect equivalent. **Keep.** |
| `rescore_xsum.py` | Re-scores stored arms on six metrics beyond ROUGE, with paired bootstrap intervals. | Replaced by the inline scorer. |
| `rescore_xsum_unieval.py` | The same for UniEval's four dimensions. | Replaced by the inline scorer. |
| `semantic_validation.py` | Tests the embedding metric against the doubt recorded in `semantic_metrics.py`'s docstring. | Keep, but see the perturbation note below. |
| `score_comet.py` | Scores COMET on the bake-off conditions. | Keep — COMET cannot join the evaluate environment (see below). |

---

## Known defects and where they stand

A review in October 2026 found the following. Recording them here rather than
in a ticket, because several are only visible by reading two files together.

**Fixed by moving to the inline scorer**

1. **Empty generations were handled two ways.** `rescore_xsum.py` scores an
   empty output 0.0; `rescore_xsum_unieval.py` dropped it. The two tables were
   documented as readable side by side and were not: the degenerate `kd2-a09`
   arm (ROUGE-1 0.015, 85/200 empty) came out at UniEval consistency 0.876,
   *above* the merge's 0.848, so the arm named in the job header as the control
   proving the metrics work did not do so. `xsum_faithfulness.py` scores an
   empty completion 0.0 for every metric, in one code path.
2. **The G-Eval digit lookup included the space token.** `" 3"` tokenises as
   `[space, "3"]` on the Gemma tokeniser, and the code took the first token, so
   every rating shared the space token's probability and all scores were pulled
   toward the middle. The new file keeps only ids that decode to the digit.
3. **Articles were paired with predictions by list position.** Any change of
   dataset order would have scored summaries against the wrong article
   silently. Inspect passes each scorer its own sample; the article comes from
   `state.metadata["document"]`.

**Fixed here**

4. `isambard/config.sh` and `isambard/93_rescore_xsum.slurm` were missing from
   `main` while four job scripts sourced them. Restored.
5. `bert-score` was imported but absent from `evaluate/pyproject.toml`.  Added.
6. `94_comet.slurm` hardcoded one user's venv path. Now `$METRICS_VENV`,
   defaulting to `metrics/.venv`.
7. `metrics/` pinned torch to cu128 while the rest of the repo uses cu130.
   Unpinned, so it resolves to the same CUDA build as every other measurement.

**Open — judgement calls, not bugs**

- **Duplicated code.** `sentences`, SummaC, G-Eval, UniEval and the bootstrap
  are copied across three legacy files and have already drifted; defect 1 is a
  consequence. The inline scorer has one copy of each. The legacy files are
  left as they were, as the record of how published numbers were produced.
- **`semantic_validation.py` and `metric_bakeoff.py` perturb differently** —
  verb list, number offsets `[7,9,11,13]` vs `[7,9,11]`, unrelated-summary
  offset 500 vs 97. Their results cannot be read across. Unify before quoting
  them together.
- **The bake-off treats real model outputs as correct.** `model_output` and
  `paraphrase_nat` are counted as faithful, but the merge hallucinates in 578
  of 1000 cases, which mixes noise into every separation score.
- **The judge is the teacher.** The G-Eval judge is `gemma-3-4b-it`, also the
  self-distillation teacher. A judge favours output that sounds like itself.
- **Generation length changed.** `xsum_task.py` generates up to 128 tokens; the
  Isambard arms used 64. `pred_words` especially is not comparable across the
  two, and neither are metrics sensitive to truncation.

---

## Environments and CUDA

Everything except COMET runs in `evaluate/`:

```bash
uv run --project evaluate python experiments/<script>.py
```

COMET keeps its own environment: `unbabel-comet` pulls pytorch-lightning and
pins transformers back to 4.x, which would downgrade the library every result
in this programme was measured under.

Both environments leave torch unpinned so that on Isambard-AI (GH200, aarch64)
they resolve to the same CUDA 13.0 build. Since October 2026 the node driver
(580.173) supports CUDA 13.0 natively, so no forward-compatibility libraries
are needed - and the old ones now *break* the GPU (CUDA error 803), because they
are older than the driver. The job scripts strip any compat directory from an
inherited `LD_LIBRARY_PATH` and assert `torch.cuda.is_available()` before
running, so a job can never fall back to the CPU in silence; see
`evaluate/evaluate.sbatch`.

`isambard/config.sh` defines `REPO_ROOT`, `DATASETS_DIR`, `MODELS_DIR`,
`RESULTS_DIR` and the HF cache. Its defaults assume the Isambard layout
(`$REPO_ROOT/../datasets`), so in a laptop clone they will not resolve — set
them in the environment or run the scripts with explicit paths.
