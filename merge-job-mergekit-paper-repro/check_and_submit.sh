#!/bin/bash
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Run: export HF_TOKEN=\"your-token\""
  exit 1
fi

echo "Checking access to epfl-llm/meditron-7b..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/models/epfl-llm/meditron-7b)
echo "HTTP status: $STATUS"

if [ "$STATUS" != "200" ]; then
  echo "Access check failed (status $STATUS) -- not submitting the job."
  echo "Check https://huggingface.co/settings/gated-repos for epfl-llm/meditron-7b."
  exit 1
fi

echo "Access confirmed. Submitting job..."
az ml job create -f job.yml --resource-group TIRE-1 --workspace-name TIRE-2 \
  --set inputs.hf_token="$HF_TOKEN"
