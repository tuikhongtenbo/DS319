#!/usr/bin/env bash
set -euo pipefail

# Clone and install the upstream LLaVA repository for inference / LoRA training.
# Usage: bash scripts/setup_llava.sh [LLAVA_DIR]
#
# LLaVA's pyproject.toml pins torch==2.1.2, which is unavailable on Python 3.12+
# and many CUDA indexes. We install a compatible torch stack first, then install
# llava with --no-deps to skip the strict pin.

LLAVA_DIR="${1:-./LLaVA}"

if [ ! -d "${LLAVA_DIR}/.git" ]; then
  git clone https://github.com/haotian-liu/LLaVA.git "${LLAVA_DIR}"
fi

cd "${LLAVA_DIR}"

echo "Installing PyTorch (skipping LLaVA's torch==2.1.2 pin)..."
pip install "torch>=2.2.0" "torchvision>=0.17.0"

echo "Installing LLaVA package (editable, no dependency resolution)..."
pip install -e . --no-deps

echo "Installing LLaVA runtime dependencies..."
pip install \
  "transformers==4.37.2" \
  "tokenizers==0.15.1" \
  "sentencepiece==0.1.99" \
  "shortuuid" \
  "accelerate==0.21.0" \
  "peft" \
  "bitsandbytes" \
  "pydantic" \
  "markdown2[all]" \
  "numpy" \
  "scikit-learn>=1.2.2" \
  "requests" \
  "httpx==0.24.0" \
  "uvicorn" \
  "fastapi" \
  "einops==0.6.1" \
  "einops-exts==0.0.4" \
  "timm==0.6.13" \
  "Pillow" \
  "tqdm"

echo ""
echo "Verifying LLaVA import..."
python -c "from llava.model.builder import load_pretrained_model; print('LLaVA OK')"

echo ""
echo "LLaVA installed at ${LLAVA_DIR}"
echo "Run generated train_llava.sh or train_spacellava.sh from DS319 outputs."
