#!/usr/bin/env bash
set -euo pipefail

# Fine-tune LLaVA (liuhaotian/llava-v1.5-7b)
# Generates a train_llava.sh script inside the output directory;
# execute that script inside the LLaVA repository root.
# Usage: bash scripts/llava_ft.sh [extra args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode train \
    --config "${ROOT_DIR}/src/configs/train_llava.yaml" \
    "$@"
