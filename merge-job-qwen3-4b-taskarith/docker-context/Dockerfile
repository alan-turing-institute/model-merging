FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir "git+https://github.com/arcee-ai/mergekit.git"
RUN pip install --no-cache-dir lm-eval pandas scipy statsmodels
