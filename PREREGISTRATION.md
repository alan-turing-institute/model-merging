# Preregistration

**What this document does:** it fixes the hypotheses, the conditions, the
outcome measures, the statistical tests and the decision rules for the
disjoint-halves merging experiment *before* the runs that are meant to settle
it. Everything already in `results/` is reclassified as pilot data by this
document and cannot be used as confirmatory evidence for anything.

**Why now.** The pipeline works, three arms exist ([crime](README.md),
[XSum](XSUM.md), [cross-task](CROSSTASK.md)), and there are eight merge methods
with configs in `merge/`. That is a large enough surface that some contrast
somewhere will look significant whether or not merging works, and the pilot
already contains one contrast (TIES beating a linear merge by 2.8pp) that would
be very easy to write up as a finding. There is currently no pre-specified
primary comparison and no measured noise floor to judge it against, so a
finding and an artefact are indistinguishable. This document is what makes the
next round of compute capable of producing an answer rather than another table.

**Status:** drafted 2026-09-09. Nothing below has been run against the
confirmatory design yet. Amendments go in [Deviations](#deviations), appended
with a date, never by editing the text above them.

## The question

Split a training set into two disjoint halves, fine-tune an expert on each half
independently, merge the two experts into one set of weights. Does the merge
recover the model you would have got by training on all the data at once?

If it does, fine-tuning parallelises across machines in a way gradient
synchronisation does not — poor man's parallelism. If it does not, the
interesting part is *how* it fails, because the failure mode is diagnostic of
what merging actually does to a task vector.

### Hypotheses

Stated directionally, because the pilot points somewhere and pretending
otherwise would be dishonest rather than neutral.

| | |
|---|---|
| **H1 (primary)** | A merge of two disjoint-half experts does **not** recover full-data performance: recovery fraction *R* < 1 (defined below). |
| **H2 (primary)** | A 0.5/0.5 linear merge does not beat the better of its own two inputs: *R* ≤ 0. |
| **H3 (secondary)** | The shortfall is concentrated in a return toward base-model behaviour — on crime, a recall collapse with precision held; on XSum, output length and sentence count drifting back toward base. |
| **H4 (secondary)** | Methods that combine deltas at full strength or by sign consensus (`task_arithmetic` w=1.0, `ties`) recover more than `linear` does. |

H1 and H2 are separate claims and the pilot separates them: linear merges
failed both, TIES failed H1 but passed H2. Reporting only one of them is how
this experiment gets mis-stated in either direction.

### Primary estimand

Per arm, the **recovery fraction**

```
R = (merge − best_half) / (full − best_half)
```

where each term is that arm's primary metric (below), `best_half` is whichever
half-data expert scored higher *on the test set in that same run*, and `full`
is the full-data model.

- `R ≥ 1` — the merge reached the full-data model. Poor man's parallelism works.
- `0 < R < 1` — merging bought something, but not the full-data model.
- `R ≤ 0` — the merge is no better than just keeping the better half and
  throwing the other one away, which makes the whole procedure pointless.

The denominator is deliberately `full − best_half` and not `full − base`.
Retention against base (which is what `experiments/crosstask_table.py` reports,
and the right choice there) flatters a half-data merge, because most of the
distance from base to full is bought by *any* fine-tuning at all: base scores
0.748 and a single half scores 0.924, so a merge that has learned nothing the
halves didn't already know still posts a retention near 0.8. `best_half` is the
honest baseline because it is the thing a practitioner would otherwise do.

`best_half` is chosen per run rather than fixed in advance, and that choice is
made on the same test set that produces the numerator — which biases *R*
downward. This is accepted, and stated here rather than discovered later: the
bias is conservative with respect to H1 and H2, both of which predict low *R*.
The `worst_half` and `mean_half` denominators are also reported so the size of
the bias is visible.

### Primary metric per arm

Fixed now, and these are the existing tooling defaults so that the choice is
not itself a degree of freedom:

| Arm | Primary metric | Source |
|---|---|---|
| Crime classification | accuracy | `evaluate/evaluate.py`; tested by `experiments/mcnemar.py` |
| XSum summarisation | ROUGE-1 F | `evaluate/evaluate_summarisation.py`; tested by `experiments/bootstrap_rouge.py` (its `--metric` default) |
| Cross-task | per-task retention against base, crime F1 and XSum ROUGE-1 | `experiments/crosstask_table.py` |

F1, precision, recall, ROUGE-2, ROUGE-L, output length and unparsed counts are
all still recorded and all still reported. They are secondary. A result that
holds on accuracy but not F1 (or vice versa) is reported as mixed, not as
whichever one came out better.

The cross-task arm keeps retention-against-base rather than *R*, because there
is no "full" model that holds both skills — nothing was trained on the union of
the two tasks. It answers a different question and is not evidence for or
against H1.

## What is pilot data

**Everything in `results/`, and everything described in the arm docs, is
pilot.** It was collected while the pipeline was being built, under conditions
that were changing, and it is used here for exactly three things: choosing the
primary metric, estimating discordance for the power calculation below, and
motivating the hypotheses. It is not evidence.

Pilot, crime arm, from the twelve JSONs in `results/` (n=1077, 0 unparsed
generations in every condition):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Full data | 0.9694 | 0.9684 | 0.9623 | 0.9653 |
| Merged, `ties` | 0.9499 | 0.9710 | 0.9140 | 0.9417 |
| Half 1 alone | 0.9313 | 0.9975 | 0.8470 | 0.9161 |
| Half 2 alone | 0.9239 | 0.9759 | 0.8491 | 0.9081 |
| Merged, `linear` / `slerp` / `task_arithmetic` | 0.9220 | 0.9925 | 0.8302 | 0.9041 |
| Merged, `dare_ties` | 0.9155 | 0.9923 | 0.8155 | 0.8953 |
| Merged, `arcee_fusion` | 0.9109 | 0.9974 | 0.8008 | 0.8884 |
| Merged, `model_stock` | 0.7837 | 0.9841 | 0.5199 | 0.6804 |
| Base, zero-shot | 0.7484 | 0.9682 | 0.4465 | 0.6112 |

Three things in that table shape the design, and one of them is a problem.

**`linear`, `slerp` and `task_arithmetic` are identical to the example.** Not
similar — the same 1077 predictions. At weight 0.5/0.5 with two models over a
shared base they are the same arithmetic, so this is a reassuring
implementation check rather than a finding, but it also means those three are
one condition and must not be counted as three. The two routes to a linear
merge also agree exactly (`gemma3-crime-merged-linear-2` and
`...-linear-2-from-lora`, i.e. merging materialised half models versus merging
adapters with mergekit's `<base>+<adapter>` syntax), which is a second free
check on the merge path.

**The recall/precision split is exactly what H3 predicts.** Every merge holds
precision at or above 0.99 and loses accuracy through recall. `model_stock`
collapses almost all the way back to base. That is under-adaptation, not
confusion.

**The pilot's own halves disagree between runs, and this must be resolved
before anything else runs.** The results in `results/` put half 1 at 0.9313 and
half 2 at 0.9239 — a 0.74pp gap, and by exact McNemar not distinguishable from
zero (26/18 discordant, p = 0.29). Earlier notes from the same day record the
same two conditions at 0.9211 and 0.9582 — a 3.7pp gap in the *other*
direction, with the full-data model at 0.9768 and the linear merge at 0.9461.
Both sets cannot describe the same artifacts. Either two different training
runs are being conflated, or a results directory was overwritten, or the
registered model versions differ between the two evaluations.

**The discrepancy cannot be resolved from the result files, and this is
settled rather than pending.** All twelve JSONs record their input only as a
local relative path — `../models/gemma3-crime-1-of-2` and so on — with no
registry name, no version and no timestamp. Those directories lived on a
compute instance and are gone. `az ml model list` can say which versions exist
and when each was created, but nothing links a given JSON to a given version,
so no audit can attribute these numbers after the fact.

The pilot is therefore **unattributable, and is discarded for every
confirmatory purpose**. Its internal consistency is a reasonable bet — all
twelve files share a single mtime and all read from local paths, which is what
one evaluation pass on one instance looks like — and that is why they are still
used for the sensitivity calculation below. They are not used for anything else,
and no number from them is repeated as a result.

The `--provenance` flag added to both evaluators after these runs
(`53a58da`) fixes this going forward: every future results JSON records which
registered version produced it. The version audit
[CROSSTASK.md](CROSSTASK.md) warns about is still required — several versions of
each name exist and some are smoke-test artifacts under production names — but
its job is choosing inputs for the confirmatory runs, not rescuing the pilot.

## Fixed design

### Conditions

Per single-task arm (crime, XSum), five conditions, all evaluated on the same
held-out test set:

| Condition | Role |
|---|---|
| base, zero-shot | floor |
| full-data expert | target |
| half-1 expert | merge input, and baseline for *R* |
| half-2 expert | merge input, and baseline for *R* |
| merge of the two halves | the experiment |

The base and both halves are non-negotiable. A merge compared only against the
full-data model cannot distinguish "merging works" from "the merge landed
between its inputs and one input was already good", which is the pilot's actual
result.

### Merge methods

**One confirmatory method per arm: `linear` at 0.5/0.5.** It is the naive thing
a practitioner would try, it is the method H2 is about, and it keeps the
confirmatory family small enough to test.

Every other method — `task_arithmetic` at w=0.5 and w=1.0, `ties`,
`dare_ties`, `slerp`, `model_stock`, `arcee_fusion` — is **exploratory**, and
so is the whole of H4. They are run, tabulated and reported with p-values, and
those p-values are descriptive: with eight methods, α=0.05 and no correction,
roughly one spurious "winner" per family is expected. The pilot's TIES result
is a hypothesis generated by this experiment, not one tested by it. Confirming
it needs its own pre-specified run with TIES named in advance as the single
confirmatory method — which is the natural follow-up and is explicitly not what
this document licenses.

### Replication, and what the unit of replication is

**Five training seeds per condition, 42–46.** Seed 42 is axolotl's default and
is what every existing run used, so the pilot supplies that arm.
`experiments/seed_sweep.sh` writes the derived configs; its default of two
extra seeds is enough to establish that noise is large and not enough to bound
it, so `SEEDS="43 44 45 46"` is the pre-specified invocation.

This is the single most important commitment in the document, because the
experiment has two independent sources of variance and the pilot only ever
addressed one of them:

| Source | What varies | How it is quantified |
|---|---|---|
| Test-set sampling | which 1077 examples | McNemar (crime) / paired bootstrap (XSum), within a run |
| Training noise | seed, data order, nondeterministic kernels | spread across the five seeds, per condition |

**A claim about merging is a claim about the procedure, not about one pair of
weights, so the unit of replication is a training run.** McNemar on a single
pair of models answers "do these two checkpoints differ on this test set",
which is a real and much narrower question. Both are reported; only the
across-seed comparison supports H1 or H2.

The five-seed budget is deliberately spent on the *halves* and their merge, not
on the full-data model. Under-powering the target condition is a real cost, and
it is accepted because H1 and H2 both concern the merge relative to its own
inputs, and because the halves are where the pilot's instability appeared.

### Everything else is held fixed

Base `google/gemma-3-4b-it`; LoRA r=16, α=32, targets
`q_proj,k_proj,v_proj,o_proj`; the committed `train/*.yaml` other than
`output_dir` and `seed`; the test sets as produced by `prepare_data.py` and
`prepare_xsum_data.py` at `XSUM_SEED=42`; and the exact registered model
version of every artifact, recorded per run.

**Registered versions are recorded before evaluation, not after.** This is the
failure mode [CROSSTASK.md](CROSSTASK.md) flags: several versions of each name
exist, some are smoke-test artifacts under production names, and merging the
wrong one silently measures a different experiment.

## Analysis plan

Fixed before the data:

1. **Provenance first.** Reconcile the pilot discrepancy above, or discard the
   pilot. Record the registered version of every artifact used.
2. **Halves against each other, before anything downstream.** Within each seed,
   half 1 vs half 2 (exact McNemar / paired bootstrap); across seeds, the
   within-half spread. If the two halves differ systematically across seeds,
   the splits are not exchangeable and every comparison downstream of them is
   confounded — stop and fix the split rather than reporting *R*.
3. **Primary test.** *R* per seed, then the mean *R* across the five seeds with
   a 95% interval from the across-seed spread. H1 is tested by whether that
   interval excludes 1; H2 by whether it excludes 0.
4. **Within-run tests, reported alongside.** Exact McNemar (binomial on
   discordant pairs, not the chi-square approximation — discordant counts here
   are 11–75, where the approximation is unreliable) for crime;
   `bootstrap_rouge.py`'s paired bootstrap, 10 000 resamples, for XSum, which
   reports an interval on the per-example difference rather than only a p-value.
5. **Multiplicity.** Two primary hypotheses on one estimand per arm:
   Holm–Bonferroni across {H1, H2} within each arm, α=0.05 family-wise. Arms
   are not corrected against each other — they are separate experiments with
   separate datasets, and each is reported on its own. Everything under
   "exploratory" is uncorrected and labelled as such wherever it appears.
6. **H3, the mechanism.** Recall at held precision (crime); `mean_pred_words`
   and `mean_pred_sentences` against the base model's values (XSum);
   `num_unparsed` and format leakage (cross-task). Read these *before* the
   headline metric — a merge that has drifted toward base shows it here first,
   and it is a more legible statement than a metric delta.
7. **Read generations.** At least 20 per condition, from the `predictions`
   field. A ROUGE gap of a few tenths is not necessarily a difference a reader
   would notice, and this is the check that catches that.

### Sensitivity

Computed from the pilot's actual discordance, not assumed.

Across the pilot's pairwise contrasts, two models disagree on **11–75 of the
1077 test examples** — and on **24–54** for the contrasts this design actually
turns on: a merge against either of its own inputs, and half 1 against half 2.
McNemar's power is set by that discordant count, not by n=1077, which makes the
effective sample far smaller than the test set looks:

| Discordant pairs | Smallest resolvable accuracy gap (α=0.05, exact) |
|---|---|
| 24 | ~1.0pp |
| 50 | ~1.5pp |
| 75 | ~1.8pp |

So **a single run resolves differences of about 1pp and up, and nothing
finer.** Pilot contrasts below that (linear vs half 2, 0.19pp; half 1 vs half 2,
0.74pp) are not small effects — they are unresolved.

The across-seed test is the binding constraint, and five seeds is few. If the
within-half SD comes in near the 3.7pp the earlier notes imply, five seeds
resolve differences of roughly 5pp in mean *R* terms and the experiment cannot
settle H1 at all. **That contingency is pre-specified rather than left to
judgement:** see the last row of the decision table.

## Decision rules

Committed in advance, so that the result is read the same way whichever way it
comes out.

| Outcome | Conclusion |
|---|---|
| Mean *R* interval excludes 1 and lies above 0 | Merging recovers some but not all of the full-data model. H1 supported, H2 rejected. The headline is the size of *R*, not its sign. |
| Mean *R* interval excludes 1 and includes or lies below 0 | Poor man's parallelism fails as tested: the merge is not better than keeping the better half. H1 and H2 both supported. |
| Mean *R* interval includes 1 | No detectable shortfall. H1 not supported — reported as such, and the sensitivity above is reported with it, since "includes 1" with five seeds is weak evidence for recovery. |
| Halves differ systematically across seeds (step 2) | Splits not exchangeable. No claim about merging. Fix the split; this design is void and gets re-registered. |
| Within-half SD large enough that the *R* interval spans both 0 and 1 | **Inconclusive, and reported as inconclusive.** Not rescued by dropping to the within-run McNemar result, which does not address training noise and would be a different claim from the one asked. The next step is more seeds or a larger test set, not a different analysis. |

**H3 is diagnostic and does not gate any of the above.** If *R* is low and the
mechanism is *not* a drift toward base, that is a more interesting result than
the confirmation would have been, and it gets reported prominently rather than
buried.

### What would make us abandon the framing

- `linear` and `task_arithmetic` w=0.5 stop agreeing exactly — the arithmetic
  says they must at 0.5/0.5 over a shared base, so a divergence is a bug
  somewhere in the merge path and invalidates the runs, not the hypothesis.
- The full-data model fails to beat both halves. Then the task is saturated or
  the training is broken, and there is no headroom for a merge to recover.
- Unparsed generations appear in the crime arm at any material rate. The pilot
  had zero across twelve conditions; a change there means the evaluation
  prompt or the parser moved, and the runs are not comparable to the pilot.

## Runs this document licenses

In order. On a GPU compute instance, under `screen`, with the instance stopped
afterwards — an A100 bills for wall-clock uptime, not GPU use.

```bash
az ml model list --resource-group tire-1 --workspace-name tire-2 -o table   # step 1: pick versions
```

Step 1 selects and records the input versions; it does not attempt to attribute
the pilot, which the section above establishes is impossible. Pass
`--provenance` to every evaluation from here on so this cannot recur.

```bash
SEEDS="43 44 45 46" experiments/seed_sweep.sh                              # step 2: the noise floor
```

```bash
SEEDS="43 44 45 46" MERGE_SEEDS=1 experiments/seed_sweep.sh                # step 3: the primary test
```

```bash
experiments/merge_methods.sh                                               # exploratory only
```

Then the analysis, from the repo root:

```bash
uv run --project evaluate python experiments/mcnemar.py "$RESULTS_DIR"
```

Step 3 is where the disk cost lives (~26GB peak, from the `--lora-merge-cache`
plus one merge output). Step 2 evaluates adapters as `base+adapter` and needs
~540MB per run. `MERGE_CACHE_DIR=/tmp/lora-merge-cache` if the root disk is the
tighter one, remembering `/tmp` is on the temp disk and does not survive an
instance stop.

Two cost items to plan for rather than discover:

- **Step 3 rebuilds the merge cache once per seed.** `seed_sweep.sh` deletes
  `models/.lora_merge_cache` after each seed's merge, so the ~17GB
  materialisation is paid four times, not once. That deletion is also what makes
  the step *correct* — see the cache hazard below — so it should not be
  optimised away without replacing it with something that invalidates the cache
  per seed.
- **`evaluate/evaluate.py` generates one example at a time, with no batching**,
  unlike `evaluate_summarisation.py`. Step 2 is ten evaluations of 1077
  examples and step 3 adds four more. Batching it first (the summarisation
  evaluator's batching was checked for batch-invariance on this model family,
  so the pattern is known-good here) is the single largest saving available on
  this design, and it changes no result.

### The merge cache is a correctness hazard, not just a disk cost

mergekit keys `--lora-merge-cache` on the *model reference* — base id plus
adapter path — and not on the weights' contents. Nothing in the pipeline links
"these adapters were retrained" to "that cache is now stale", so a cache built
from earlier adapters at the same paths is silently reused, and the merge
combines the **old** weights while every log line names the new ones. A wrong
result that looks completely clean is the worst failure mode this experiment
has.

Two consequences for the runs above:

- **Step 3 is safe, by construction rather than by design.** Each seed's
  adapters live at seed-specific paths, so the cache keys differ, and the script
  deletes the cache after each seed anyway.
- **`experiments/merge_methods.sh` is not.** It reads the unseeded paths
  `models/gemma3-crime-{1,2}-of-2-lora` and deliberately reuses one persistent
  cache across methods — which is the right optimisation given a valid cache and
  a silent wrong answer given a stale one. **Delete the cache by hand before
  running it**, every time, until invalidation is wired in. Note also that
  `run_xsum_pipeline.sh` treats a non-empty cache as costing 0GB in its preflight
  check, so a stale cache actively *relaxes* the space guard rather than raising
  a flag.

**No XSum results are in hand.** The arm's first full run trained, registered,
converted and merged successfully, then lost its five evaluations to a chain of
disk-space and path-resolution failures; whether they were completed on a later
attempt is unknown from here — no `gemma3-xsum-*.json` exists in the local
clone, and `$RESULTS_DIR` lives on the instance's `cloudfiles` share, which is
not mounted locally. **Check the share before assuming they need rerunning.**

Either way the arm has no pilot data available to this document, which means no
discordance estimate and no sensitivity figure of its own; the numbers in the
sensitivity section are the crime arm's and do not transfer to ROUGE. Two prerequisites before that arm counts as
confirmatory rather than pilot: complete one full evaluation pass, and write it
a seed-sweep script (it has none).

Nothing above depends on the XSum arm, and it stays in this document because
H3's mechanism claim is much more legible on a generation task. It is
sequenced second deliberately.

## Deviations

Every departure from the above gets a dated entry here, with what changed and
why. An empty section means the design was followed. Amendments made *before*
the corresponding data exists are still amendments and still go here — the
point is the record, not the blame.

*(none yet)*
