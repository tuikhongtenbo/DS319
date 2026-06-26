#!/usr/bin/env bash
set -euo pipefail

# GPT-4o 1-shot inference via OpenAI API
# Requires: --api_key YOUR_OPENAI_API_KEY
# Usage: bash scripts/gpt_infer_1_shot.sh --api_key sk-... --out_results ./results/gpt_1shot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode infer \
    --config "${ROOT_DIR}/src/configs/train_gpt.yaml" \
    --shots 1 \
    "$@"
