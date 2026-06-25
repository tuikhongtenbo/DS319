#!/usr/bin/env bash
set -euo pipefail

# Pre-download a Hugging Face model with stable HTTP settings.
# Usage: bash scripts/download_hf_model.sh REPO_ID [LOCAL_DIR]
#
# Examples:
#   bash scripts/download_hf_model.sh liuhaotian/llava-v1.5-7b
#   bash scripts/download_hf_model.sh bczhou/SpaceLLaVA ./models/SpaceLLaVA
#   bash scripts/download_hf_model.sh MAGAer13/mplug-owl-llama-7b-pt

REPO_ID="${1:?Usage: download_hf_model.sh REPO_ID [LOCAL_DIR]}"
LOCAL_DIR="${2:-}"

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

if [ -n "${LOCAL_DIR}" ]; then
  huggingface-cli download "${REPO_ID}" --local-dir "${LOCAL_DIR}" --local-dir-use-symlinks False
else
  huggingface-cli download "${REPO_ID}"
fi

echo "Downloaded ${REPO_ID}"
