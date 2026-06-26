#!/usr/bin/env bash
set -euo pipefail

# Inference with SpaceLLaVA via HuggingFace
# Usage: bash scripts/spacellava_infer.sh [extra args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode infer \
    --config "${ROOT_DIR}/src/configs/train_spacellava.yaml" \
    "$@"
