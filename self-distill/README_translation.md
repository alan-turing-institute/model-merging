# Self-distillation on Chinese-to-English translation

We perform self-distillation using the kd plugin in Axolotl to train Gemma 3 to translate Chinese sentences into English, using (some of) the WMT News Commentary corpus.  This follows the same pipeline as XSum (see [`README.md`](README.md)); only the data preparation and the config differ.

The aim is to get an output which is

- a faithful English translation of the whole sentence
- the translation only, i.e. no "Here is the translation: ...", no notes, no pinyin

## Pipeline

#### Data preparation

We use WMT's own release of News Commentary v18, not the OPUS copy.  In the OPUS zh-en alignment most English pairs are matched with a Chinese paragraph containing an extra sentence, so a gold reference would teach the model to drop content.  The WMT release is aligned one sentence to one sentence.

```bash
uv run python prepare_data_NewsCommentary_zh_en.py \
  https://data.statmt.org/news-commentary/v18/training/news-commentary-v18.en-zh.tsv.gz \
  ../datasets/news_commentary_zh_en_prompts  # build prompt variants
```

The first argument is the WMT TSV, either a URL or a local `.tsv`/`.tsv.gz`; the second is where to save the prompts.  The script:

- cleans the corpus: drops untranslated Project Syndicate boilerplate, duplicates, very short or long sentences and likely misalignments (`--min-words`, `--max-words`, `--min-ratio`, `--max-ratio`)
- splits train/validation/test by article, so no test sentence shares an article with a training sentence
- samples 10k/1k/1k rows (`--train-size`, `--validation-size`, `--test-size`), with the validation and test rows drawn from ~250 held-out articles each (`--holdout-factor`)

As for XSum, it outputs the teacher prompt, a short and a long student prompt, and the gold English translation as `reference`.  Unlike the XSum teacher prompt, these prompts have not been through a bake-off.

#### Teacher training

The same sbatch script as for XSum; the defaults fit, since the English references are at most ~113 tokens.

```bash
sbatch generate_teacher_logprobs.sbatch google/gemma-3-4b-it \
  $PROJECTDIR/$USER/model-merging/datasets/news_commentary_zh_en_prompts \
  $PROJECTDIR/$USER/model-merging/datasets \
  gemma3_4b_nc_zh_en_kd_data
```

Add `--targets gold` (and a `_gold` prefix) to train against the reference translations instead of the teacher's own, or use `google/gemma-3-27b-it` as the teacher.

#### Student training

```bash
sbatch train_kd.sbatch self-distillation_nc_zh_en.yaml \
  $PROJECTDIR/$USER/model-merging/models/gemma3_4b_nc_zh_en_short \
  $PROJECTDIR/$USER/model-merging/datasets/gemma3_4b_nc_zh_en_kd_data_short
```

`self-distillation_nc_zh_en.yaml` has exactly the training settings of `self-distillation_xsum.yaml`, so the two tasks differ only in their data.  `train_kd.sbatch` is the same script for both.

#### Evaluation

The Inspect AI task is [`../evaluate/translation_task.py`](../evaluate/translation_task.py).  It runs in two steps, from `../evaluate`: generation scored by chrF, BLEU and behaviour checks, then COMET-22 and CometKiwi added to the same log from a separate environment (`unbabel-comet` needs older `transformers` and `numpy`).

```bash
sbatch evaluate.sbatch translation ../models/gemma3_4b_nc_zh_en_short -T prompt_variant=short
sbatch comet_score.sbatch logs/<log>.eval
```

`prompt_variant` must match the prompt the model was trained on.  The behaviour checks were refined on the validation split from base-model and teacher outputs and are now frozen (version 2), so report results on `test` (the default).  Logs scored with the checks' version are marked `behaviour_checks_version` in their metadata.
