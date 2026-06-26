# SpatialMQA Reimplementation

Clean reimplementation of [SpatialMQA](https://github.com/liuziyan/SpatialMQA) aligned with the original training and inference behavior. This repository provides a unified dispatcher (`main.py`) over model-specific trainers and inference scripts.

## Supported Models

| Task | Model | Setting |
|------|-------|---------|
| Finetune | BLIP-vqa-base | Full |
| Finetune | BLIP2-opt-2.7B | LoRA |
| Finetune | LLaVA-1.5-7B | LoRA (external script) |
| Finetune | SpaceLLaVA | LoRA (external script) |
| Inference | BLIP-vqa-base | Zero-shot |
| Inference | BLIP2-opt-2.7B | Zero-shot |
| Inference | LLaVA-1.5-7B | Zero-shot / LoRA |
| Inference | SpaceLLaVA | Zero-shot / LoRA |
| Inference | GPT-4o | 0-shot / 1-shot |
| Inference | Qwen-VL | 0-shot / 1-shot |

## Architecture

- `src/trainers/` — model-specific training loops or external script generators
- `src/inference/` — model-specific inference logic
- `src/datasets/preprocessing.py` — shared prompts and path helpers
- `src/datasets/collator.py` — batch padding for BLIP-family trainers
- `src/metrics/metrics.py` — macro P/R/F1/Acc aligned with original eval
- `main.py` — CLI dispatcher for train / infer / eval
- `scripts/*.sh` — ready-to-run shortcuts for every model and task

## Getting Started

### 1. Installation

```bash
git clone https://github.com/tuikhongtenbo/DS319.git
cd DS319
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Then install dependencies based on your task:

**For training BLIP / BLIP-2:**

```bash
pip install -r src/requirements/requirement_blip.txt
```

**For training or inference with LLaVA / SpaceLLaVA:**

```bash
# Install torch FIRST (choose your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install LLaVA requirements
pip install -r src/requirements/requirement_llava.txt
```

For LLaVA / SpaceLLaVA fine-tuning, also clone the upstream repo:

```bash
bash scripts/setup_llava.sh /workspace/LLaVA
```

For mPLUG-Owl, clone [X-PLUG/mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) and add its `mPLUG-Owl/` directory to `PYTHONPATH`.

### 2. Dataset Preparation

Bundled splits live in `src/datasets/data/`:

```text
src/datasets/data/
├── train.jsonl   # 3780 samples
├── dev.jsonl     # 536 samples
└── test.jsonl    # 1076 samples
```

Download COCO2017 test images:

```bash
bash scripts/download_coco.sh
```

Expected image layout:

```text
data/images/COCO2017/
├── 000000000933.jpg
└── ...
```

### 3. Fine-Tuning

Run fine-tuning for each model with a single command:

```bash
# BLIP (full fine-tune)
bash scripts/blip_ft.sh --out_checkpoint ./outputs/blip_checkpoints --out_results ./outputs/blip_logs

# BLIP-2 (LoRA)
bash scripts/blip2_ft.sh --out_checkpoint ./outputs/blip2_checkpoints --out_results ./outputs/blip2_logs

# LLaVA (LoRA — generates external script, run inside LLaVA repo)
bash scripts/llava_ft.sh --out_checkpoint ./outputs/llava_checkpoints --out_results ./outputs/llava_logs
# then: bash outputs/llava_checkpoints/train_llava.sh  (inside LLaVA repo)

# SpaceLLaVA (LoRA — generates external script, run inside SpaceLLaVA repo)
bash scripts/spacellava_ft.sh --out_checkpoint ./outputs/spacellava_checkpoints --out_results ./outputs/spacellava_logs
# then: bash outputs/spacellava_checkpoints/train_spacellava.sh  (inside SpaceLLaVA repo)
```

**Logs saved under `--out_results`:**
- `losses.json` — training loss per step
- `dev_loss.json` — eval loss per epoch
- `log.json` — train loss + eval loss + lr per epoch
- `last_dev_metric.json` — best eval loss at end of training

### 4. Inference

Run inference for each model with a single command:

```bash
# BLIP
bash scripts/blip_infer.sh --out_results ./outputs/blip_results

# BLIP-2
bash scripts/blip2_infer.sh --out_results ./outputs/blip2_results

# LLaVA (HuggingFace)
bash scripts/llava_infer.sh --out_results ./outputs/llava_results

# LLaVA (vLLM — fast)
bash scripts/llava_infer_vllm.sh --out_results ./outputs/llava_results

# SpaceLLaVA (HuggingFace)
bash scripts/spacellava_infer.sh --out_results ./outputs/spacellava_results

# SpaceLLaVA (vLLM — fast)
bash scripts/spacellava_infer_vllm.sh --out_results ./outputs/spacellava_results

# GPT-4o 0-shot
bash scripts/gpt_infer_0_shot.sh --api_key YOUR_OPENAI_API_KEY --out_results ./outputs/gpt_0shot

# GPT-4o 1-shot
bash scripts/gpt_infer_1_shot.sh --api_key YOUR_OPENAI_API_KEY --out_results ./outputs/gpt_1shot

# Qwen-VL 0-shot (DashScope API)
bash scripts/qwen_infer_0_shot.sh --api_key YOUR_DASHSCOPE_API_KEY --out_results ./outputs/qwen_0shot

# Qwen-VL 1-shot (DashScope API)
bash scripts/qwen_infer_1_shot.sh --api_key YOUR_DASHSCOPE_API_KEY --out_results ./outputs/qwen_1shot
```

**All inference scripts** accept these common arguments:

| Argument | Description |
|----------|-------------|
| `--jsonl_dir` | Path to dataset directory (default: `src/datasets/data`) |
| `--image_dir` | Path to image directory (default: `data/images/COCO2017`) |
| `--out_results` | Output directory for predictions |
| `--shots` | Number of shots (0 or 1) — auto-set by script for GPT/Qwen |

Predictions are saved to `{out_results}/predictions.jsonl`.

### 5. Evaluation

```bash
python main.py \
    --mode eval \
    --out_results ./outputs/blip2_results
```

Metrics are printed and saved to `{out_results}/metrics.json`.

## Notes

- BLIP2 training uses manual cross-entropy with `ignore_index=1` and early stopping on dev loss, matching the original repo.
- LLaVA / SpaceLLaVA finetuning depends on upstream repositories; DS319 generates the required data files and shell scripts.
- LLaVA / SpaceLLaVA inference uses the exact prompt and conversation templates from SpatialMQA:
  - LLaVA: `"Image: <image>, Question: ..., Options: ... Output:"` with `llava_v1` / `chatml_direct` template.
  - SpaceLLaVA: `"Question: ... Options: ... Answer:"` with the same template auto-detection.
- LoRA weights for inference are loaded from `{out_checkpoint}/best_model` or `{out_checkpoint}/saved_model`, matching the original repo's `model.load_adapter` behavior.
- If GPU memory is limited, pass `--batch_size 1` or `--batch_size 2` to the ft scripts.
- Large model downloads (LLaVA, ~10 GB) disable XET automatically in `main.py`. If a download fails mid-way, remove the partial cache entry and retry:

```bash
rm -rf ~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b
bash scripts/download_hf_model.sh liuhaotian/llava-v1.5-7b
df -h /workspace   # ensure >15 GB free
```
