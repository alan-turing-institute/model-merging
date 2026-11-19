# Gemma 3 4B base model, evaluated again

The base model evaluated on crime and XSum (long prompt) on 6 November 2026 with
this branch's code on Isambard, to check that evaluations from our runs can be
pooled with main's (`../crime_data_parallel_gemma/base.json`,
`../xsum_data_parallel_gemma/base.json`). Crime is identical to main's (accuracy
0.755); XSum's ROUGE-1 is 0.2460 against main's 0.2469, from the same 1,000
prompts (`../xsum_prompts_long.json`). `index.tsv` names each file's log.
