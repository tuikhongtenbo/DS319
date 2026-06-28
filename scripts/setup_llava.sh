#!/usr/bin/env bash
set -euo pipefail

# Clone and install the upstream LLaVA repository for inference / LoRA training.
# Also downloads pre-trained models and installs DeepSpeed.
#
# Usage: bash scripts/setup_llava.sh [LLAVA_DIR]
#
# LLaVA's pyproject.toml pins torch==2.1.2, which is unavailable on Python 3.12+
# and many CUDA indexes. We install a compatible torch stack first, then install
# llava with --no-deps to skip the strict pin.

LLAVA_DIR="${1:-/workspace/LLaVA}"

if [ ! -d "${LLAVA_DIR}/.git" ]; then
  echo "Cloning LLaVA repository to ${LLAVA_DIR}..."
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
  "accelerate==0.27.2" \
  "peft==0.9.0" \
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
  "tqdm" \
  "protobuf"

echo ""
echo "Installing DeepSpeed for LoRA training..."
pip install deepspeed

echo ""
echo "Verifying LLaVA import..."
python -c "from llava.model.builder import load_pretrained_model; print('LLaVA OK')"

echo ""
echo "Verifying DeepSpeed..."
python -c "import deepspeed; print(f'DeepSpeed OK: {deepspeed.__version__}')"

# ── Download pre-trained models ─────────────────────────────────────────
echo ""
echo "============================================"
echo " Downloading pre-trained models"
echo "============================================"

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

echo ""
echo "Downloading LLaVA-v1.5-7b..."
huggingface-cli download liuhaotian/llava-v1.5-7b

echo ""
echo "Downloading SpaceLLaVA (remyxai/SpaceLLaVA)..."
huggingface-cli download remyxai/SpaceLLaVA

echo ""
echo "Downloading CLIP vision tower (openai/clip-vit-large-patch14-336)..."
huggingface-cli download openai/clip-vit-large-patch14-336

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo "LLaVA installed at: ${LLAVA_DIR}"
echo ""
echo "To fine-tune LLaVA:"
echo "  bash scripts/llava_ft.sh --out_checkpoint ./outputs/llava_checkpoints --out_results ./outputs/llava_logs"
echo ""
echo "To fine-tune SpaceLLaVA:"
echo "  bash scripts/spacellava_ft.sh --out_checkpoint ./outputs/spacellava_checkpoints --out_results ./outputs/spacellava_logs"
