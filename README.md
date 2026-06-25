# SpatialMQA Reimplementation

Clean reimplementation of [SpatialMQA](https://github.com/liuziyan/SpatialMQA) aligned with the original training and inference behavior. This repository provides a unified dispatcher (`main.py`) over model-specific trainers and inference scripts.

## Supported Models

| Task | Model | Setting |
|------|-------|---------|
| Inference | BLIP2-opt-2.7B | Zero-shot / LoRA |
| Inference | InstructBLIP-3B | Zero-shot |
| Inference | mPLUG-Owl-7B | Zero-shot / LoRA |
| Inference | LLaVA1.5-7B | Zero-shot / LoRA |
| Inference | SpaceLLaVA | Zero-shot / LoRA |
| Inference | GPT-4o | 0-shot / 1-shot |
| Finetune | BLIP-vqa-base | Full |
| Finetune | BLIP2-opt-2.7B | LoRA (in-repo) |
| Finetune | LLaVA1.5-7B | LoRA (external script) |
| Finetune | SpaceLLaVA | LoRA (external script) |

Deprecated but still available: IDEFICS, InstructBLIP finetune, Qwen.

## Architecture

- `src/trainers/` — model-specific training loops or external script generators
- `src/inference/` — model-specific inference logic
- `src/datasets/preprocessing.py` — shared prompts and path helpers
- `src/datasets/collator.py` — batch padding for BLIP-family trainers
- `src/metrics/metrics.py` — macro P/R/F1/Acc aligned with original eval
- `main.py` — CLI dispatcher for train / infer / eval

## Getting Started

### 1. Installation

```bash
git clone https://github.com/tuikhongtenbo/DS319.git
cd DS319
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For LLaVA / SpaceLLaVA, install from the upstream repo (do **not** rely on `pip install llava` alone — it pins `torch==2.1.2`, which is unavailable on Python 3.12+):

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

**BLIP2 LoRA (in-repo):**

```bash
python main.py \
    --mode train \
    --config src/configs/train_blip2.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_checkpoint ./outputs/blip2_checkpoints \
    --out_results ./outputs/blip2_logs \
    --batch_size 8
```

- Best checkpoint: `--out_checkpoint/best_model` (selected by lowest dev loss)
- Logs: `losses.json`, `dev_loss.json`, `log.json` under `--out_results`

**BLIP full fine-tune:**

```bash
python main.py \
    --mode train \
    --config src/configs/train_blip.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_checkpoint ./outputs/blip_checkpoints \
    --out_results ./outputs/blip_logs
```

**LLaVA / SpaceLLaVA (external DeepSpeed):**

```bash
python main.py \
    --mode train \
    --config src/configs/train_llava.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_checkpoint ./outputs/llava_checkpoints \
    --out_results ./outputs/llava_logs
```

Then run the generated script inside your LLaVA checkout:

```bash
bash outputs/llava_checkpoints/train_llava.sh
```

### 4. Inference

**Open-source model:**

```bash
python main.py \
    --mode infer \
    --config src/configs/train_blip2.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_checkpoint ./outputs/blip2_checkpoints \
    --out_results ./outputs/blip2_results
```

**GPT-4o zero-shot:**

```bash
python main.py \
    --mode infer \
    --config src/configs/train_gpt.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_results ./outputs/gpt4o_results \
    --api_key YOUR_API_KEY \
    --shots 0
```

**GPT-4o one-shot:**

```bash
python main.py \
    --mode infer \
    --config src/configs/train_gpt.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_results ./outputs/gpt4o_results \
    --api_key YOUR_API_KEY \
    --shots 1
```

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
- LLaVA / SpaceLLaVA / mPLUG-Owl finetuning still depends on upstream repositories; DS319 generates the required data files and shell scripts.
- If GPU memory is limited, reduce `--batch_size` to `1` or `2`.
