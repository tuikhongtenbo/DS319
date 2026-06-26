#!/usr/bin/env bash
set -euo pipefail

# Qwen-VL 0-shot inference via DashScope API
# Requires: --api_key YOUR_DASHSCOPE_API_KEY
# Usage: bash scripts/qwen_infer_0_shot.sh --api_key sk-... --out_results ./results/qwen_0shot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode infer \
    --config "${ROOT_DIR}/src/configs/train_gpt.yaml" \
    --shots 0 \
    "$@"
