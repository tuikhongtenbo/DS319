#!/usr/bin/env bash
set -euo pipefail

# Fine-tune BLIP-2 (Salesforce/blip2-opt-2.7b)
# Usage: bash scripts/blip2_ft.sh [extra args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode train \
    --config "${ROOT_DIR}/src/configs/train_blip2.yaml" \
    "$@"
