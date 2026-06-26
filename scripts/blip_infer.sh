#!/usr/bin/env bash
set -euo pipefail

# Inference with BLIP (Salesforce/blip-vqa-base)
# Usage: bash scripts/blip_infer.sh [extra args...]
#   e.g.  bash scripts/blip_infer.sh --out_results ./results/blip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode infer \
    --config "${ROOT_DIR}/src/configs/train_blip.yaml" \
    "$@"
