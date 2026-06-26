#!/usr/bin/env bash
set -euo pipefail

# Fine-tune SpaceLLaVA
# Generates a train_spacellava.sh script inside the output directory;
# execute that script inside the SpaceLLaVA repository root.
# Usage: bash scripts/spacellava_ft.sh [extra args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

python "${ROOT_DIR}/main.py" \
    --mode train \
    --config "${ROOT_DIR}/src/configs/train_spacellava.yaml" \
    "$@"
