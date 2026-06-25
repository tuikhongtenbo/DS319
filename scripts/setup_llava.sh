#!/usr/bin/env bash
set -euo pipefail

# Clone and install the upstream LLaVA repository for external LoRA training.
# Run from a directory where you want the LLaVA checkout to live.

LLAVA_DIR="${1:-./LLaVA}"

if [ ! -d "${LLAVA_DIR}" ]; then
  git clone https://github.com/haotian-liu/LLaVA.git "${LLAVA_DIR}"
fi

cd "${LLAVA_DIR}"
pip install -e .
pip install -r requirements.txt

echo "LLaVA installed at ${LLAVA_DIR}"
echo "Run generated train_llava.sh or train_spacellava.sh from this repository."
