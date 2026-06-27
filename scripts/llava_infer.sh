#!/usr/bin/env bash
set -euo pipefail

# Inference with LLaVA via vLLM
# Usage: bash scripts/llava_infer.sh [--vllm_host HOST] [extra args...]
#
# Options:
#   --vllm_host HOST   vLLM server URL (e.g., http://localhost:8000)
#                      If not provided, will create vLLM LLM instance directly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode infer \
    --config "${ROOT_DIR}/src/configs/train_llava.yaml" \
    "$@"
