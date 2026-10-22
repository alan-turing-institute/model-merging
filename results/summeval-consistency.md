# Our SummEval results against the paper (Fabbri et al., 2021, TACL)

Data: the release's 1,600 rated summaries (16 systems x 100 CNN/DailyMail test
articles) plus the original references; scripts `evaluate/summeval_meta.py`
(comparison) and `evaluate/summeval_params.py` (metric settings, output in
`summeval-params.md`). Correlations are Kendall tau over the 16 systems against
the expert means, as in the paper's Table 2. Categories for correlations: same
(|diff| <= 0.10), similar (same sign, <= 0.20), same direction but different
strength (> 0.20), different (sign differs, one side near zero), opposite (sign
differs, both |tau| >= 0.10).

## Same

- **Expert ratings.** Recomputed from the release, all 16 systems match Table 3
  to within 0.013 on every dimension (rounding): the data are joined correctly.
- **50 of 88 metric-dimension correlations** are within 0.10 of Table 2 with our
  default settings, among them every extractiveness statistic that matters:
  density vs consistency 0.60 (paper 0.65), compression vs consistency -0.43
  (-0.43), novel bi-/tri-grams vs consistency -0.52/-0.55 (-0.54/-0.57), and all
  four chrF dimensions and three of four length dimensions.
- **The paper's findings, reproduced:** lexical metrics are weakest on coherence;
  consistency follows extractiveness (density up, novel n-grams down); ROUGE-L
  tracks the experts worse than ROUGE-1/2; the pretrained systems (BART, T5,
  Pegasus) and LEAD-3 rate highest.

## Similar or comparable once the settings match the paper's

| Table 4 value (paper vs recomputed on the 100 articles) | our default | matching the paper | what differs |
|---|---|---|---|
| ROUGE-1/2/L | MAD 0.017-0.021 | 0.012-0.013 | ROUGE-L must be summary-level (sentence-split, = ROUGE-Lsum; plain ROUGE-L 0.093); no stemming is a little closer |
| chrF | 0.020 | 0.020 | same implementation and settings |
| METEOR | 0.196 | 0.021 | the paper's Java METEOR 1.5 defaults (alpha .85, beta .20, gamma .60) penalise fragmentation far harder than nltk's (beta 3) |
| BERTScore | 0.249 | 0.039 | the paper's Table 4 values are baseline-rescaled, although the toolkit's default is not |
| BLEU | 4.48 | 1.06 | case: 13 of 15 compared systems output lowercase text; lowercased BLEU matches |

The residuals that remain (about 0.01 on ROUGE and chrF, 1 BLEU point) are the
expected sampling difference: Table 4 is the full CNN/DailyMail test set, ours
the 100 annotated articles. System rankings agree at tau 0.47-0.89.

## Different, and explained by settings

- **ROUGE vs consistency.** With the original reference only, ROUGE-1/2/3/4 hardly
  correlate with expert consistency (0.02-0.10; paper 0.53-0.71). Scored against
  the 11 references averaged, as the paper did for Table 2, ROUGE-1/2 give 0.52
  and 0.60 (paper 0.53 and 0.59), and fluency 0.53/0.49 (0.52/0.48). The number
  of references, not the implementation, is the difference.
- **METEOR and BERTScore vs consistency** move the same way with 11 references:
  METEOR (1.5-like) 0.63 (paper 0.63), BERTScore-f 0.03 (0.04).

## Different or opposite, not explained

Seven correlations differ with one side near zero and four have opposite signs:
novel unigrams vs coherence and relevance (ours -0.27/-0.40, paper +0.15/+0.10),
coverage vs coherence (+0.28 vs -0.13) and relevance (+0.45 vs -0.03), and
single-reference ROUGE-L vs consistency (-0.15 vs +0.15). Likely causes, not
tested: our regex tokenisation against the paper's spaCy, our articles cut to
1,600 tokens (coverage counts only fragments found in that text), and noise -
with 16 systems, moving one or two systems changes tau by 0.1-0.2.

## Our LLM judge against the paper's metrics (not in the paper)

- System level, the Qwen3-32B judge beats every metric in Table 2 on coherence
  (0.64 vs best 0.40) and relevance (0.86 vs 0.59), and trails the best on
  consistency (0.44 vs 0.71) and fluency (0.49 vs 0.61).
- Summary level (per article across systems), it is the best on all four
  dimensions; on consistency 0.59 against at most 0.34 for any lexical metric.
- **Opposite to the experts on the references:** the paper's experts rated the
  CNN/DailyMail references poorly on coherence (3.26) and relevance (3.77); the
  judge rates them high (4.90, 4.90). On consistency both rate them below most
  systems (experts 4.47, judge 4.34). The judge is lenient on coherence and does
  not penalise the bullet-point concatenation the experts did.
