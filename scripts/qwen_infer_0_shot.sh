#!/usr/bin/env bash
set -euo pipefail

# Qwen-VL 0-shot inference via DashScope API
# Requires: QWEN_API_KEY in the environment, or pass --api_key YOUR_DASHSCOPE_API_KEY
# Usage: QWEN_API_KEY=sk-... bash scripts/qwen_infer_0_shot.sh --out_results ./results/qwen_0shot
# Optional: pass --model_name qwen3.5-flash to override the config model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode infer \
    --config "${ROOT_DIR}/src/configs/train_qwen.yaml" \
    --shots 0 \
    --num_workers 4 \
    --out_results ./outputs/qwen_0shot \
    "$@"
