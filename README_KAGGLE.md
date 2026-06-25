# SpatialMQA (Kaggle Edition)

Guide for running SpatialMQA on Kaggle Notebooks with P100 or T4 GPUs.

## Architecture

Each model has its own trainer in `src/trainers/` and inference script in `src/inference/`. The dispatcher `main.py` routes commands based on `--config`.

## Setup

1. Create a Kaggle Notebook with GPU enabled.
2. Enable Internet access.
3. Clone or attach this repository.
4. Install dependencies and download images.

```python
!git clone https://github.com/tuikhongtenbo/DS319.git
%cd DS319
!pip install -r requirements.txt

!bash scripts/download_coco.sh
```

## Fine-Tuning

```python
!python main.py \
    --mode train \
    --config src/configs/train_blip2.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_checkpoint /kaggle/working/outputs/blip2_checkpoints \
    --out_results /kaggle/working/outputs/blip2_logs \
    --batch_size 8
```

Best checkpoint is saved to `/kaggle/working/outputs/blip2_checkpoints/best_model`.

## Inference

```python
!python main.py \
    --mode infer \
    --config src/configs/train_blip2.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_checkpoint /kaggle/working/outputs/blip2_checkpoints \
    --out_results /kaggle/working/outputs/blip2_results
```

## GPT-4o on Kaggle

Store your API key in Kaggle Secrets, then:

```python
import os

!python main.py \
    --mode infer \
    --config src/configs/train_gpt.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_results /kaggle/working/outputs/gpt4o_results \
    --api_key {os.environ["OPENAI_API_KEY"]} \
    --shots 0
```

## Evaluation

```python
!python main.py \
    --mode eval \
    --out_results /kaggle/working/outputs/blip2_results
```

Metrics are written to `/kaggle/working/outputs/blip2_results/metrics.json`.

## LLaVA External Training

```python
!python main.py \
    --mode train \
    --config src/configs/train_llava.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_checkpoint /kaggle/working/outputs/llava_checkpoints \
    --out_results /kaggle/working/outputs/llava_logs
```

Then run the generated script inside an LLaVA checkout using `scripts/setup_llava.sh`.
