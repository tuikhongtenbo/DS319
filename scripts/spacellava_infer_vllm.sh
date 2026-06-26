#!/usr/bin/env bash
set -euo pipefail

# Inference with SpaceLLaVA via vLLM (fast)
# Requires: pip install -r src/requirements/requirement_vllm.txt
# Usage: bash scripts/spacellava_infer_vllm.sh [extra args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode infer \
    --config "${ROOT_DIR}/src/configs/train_spacellava.yaml" \
    "$@"
