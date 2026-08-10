FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir "git+https://github.com/arcee-ai/mergekit.git"
RUN pip install --no-cache-dir lm-eval pandas scipy statsmodels
# huggingface_hub>=1.16 tightened its hf:// URI parser to require
# 'namespace/name' repo ids, which broke `datasets`' loader for legacy
# bare-slug canonical datasets (e.g. "medmcqa", "squad") -- raises
# HfUriError before any eval even starts. Pin below that regression.
RUN pip install --no-cache-dir "huggingface_hub<1.16"
# datasets>=4.0 removed support for script-based dataset loaders entirely
# (not just gated behind trust_remote_code) -- pubmedqa's underlying
# dataset (bigbio/pubmed_qa) still ships a pubmed_qa.py builder script,
# so it hard-fails on any datasets>=4.0. Pin below that removal.
RUN pip install --no-cache-dir "datasets<4.0.0"
