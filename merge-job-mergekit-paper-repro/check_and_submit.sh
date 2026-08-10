#!/bin/bash
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Run: export HF_TOKEN=\"your-token\""
  exit 1
fi

case "$HF_TOKEN" in
  *your*token*|*xxx*|*placeholder*|*REPLACE*)
    echo "HF_TOKEN still looks like a placeholder (\"$HF_TOKEN\"), not a real token. Set your actual token first."
    exit 1
    ;;
esac

echo "Checking access to epfl-llm/meditron-7b..."
# Note: /api/models/{repo} returns 200 for anyone, authenticated or not --
# it's just public metadata and never actually checks gated access. The
# real gate is on the file-resolve endpoint mergekit/transformers actually
# hits when downloading the config, so check that instead.
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/epfl-llm/meditron-7b/resolve/main/config.json)
echo "HTTP status: $STATUS"

if [ "$STATUS" != "200" ]; then
  echo "Access check failed (status $STATUS) -- not submitting the job."
  echo "Check https://huggingface.co/settings/gated-repos for epfl-llm/meditron-7b."
  exit 1
fi

echo "Access confirmed. Submitting job..."
az ml job create -f job.yml --resource-group TIRE-1 --workspace-name TIRE-2 \
  --set inputs.hf_token="$HF_TOKEN"
