# SpatialMQA Reimplementation

This repository is a clean, production-ready reimplementation of SpatialMQA. It provides a unified pipeline for fine-tuning and inferencing both open-source models (BLIP, BLIP2, InstructBLIP, mPLUG-Owl, IDEFICS, LLaVA, SpaceLLaVA) and API-based models (GPT-4, Qwen).

### 🛠️ Decoupled Architecture (Model-Specific Scripts)
To prevent cross-compatibility errors (where a certain model depends on specific tokens, input/output structures, or training wrappers), this codebase is **fully decoupled**:
- **`src/trainers/`**: Houses independent, model-specific training loops (e.g. `train_blip2.py`, `train_idefics.py`). Each script handles its own Dataset format, collator, and training loop.
- **`src/inference/`**: Houses independent, model-specific inference logic (e.g. `inference_blip2.py`, `inference_llava.py`).
- **`main.py`**: Acts as a thin dispatcher that parses CLI arguments and dynamically routes the task to the corresponding model-specific script. Giao diện câu lệnh chạy (`main.py`) không thay đổi.

## 🚀 Getting Started (Local GPU)

### 1. Installation

First, clone this repository and set up a virtual environment:
```bash
git clone <repo_url>
cd DS319
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Next, install the core dependencies. Note that different models have specific requirements. We have provided requirement files in `src/requirements/`.

```bash
# Example for BLIP / BLIP2
pip install -r src/requirements/requirement_blip.txt

# Example for LLaVA
pip install -r src/requirements/requirement_llava.txt
```

### 2. Dataset Preparation

Ensure your dataset structure looks like this. You can download the COCO2017 test images using the provided script or manually:

```bash
mkdir -p data/images
cd data/images
wget http://images.cocodataset.org/zips/test2017.zip
unzip -q test2017.zip
mv test2017 COCO2017
rm test2017.zip
cd ../..
```

Your final folder structure should be:
```text
data/
├── images/
│   └── COCO2017/
│       ├── image1.jpg
│       └── ...
└── dataset/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

### 3. Fine-Tuning

We provide a unified entrypoint script `main.py` for training. 

```bash
python main.py \
    --mode train \
    --config src/configs/train_blip2.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data \
    --out_checkpoint ./outputs/blip2_checkpoints \
    --out_results ./outputs/blip2_results \
    --batch_size 8
```
- During training, the best model will be saved at `--out_checkpoint/best_model` based on the highest dev accuracy.
- Training metrics (`losses.json`, `log.json`, `dev_acc.json`) are saved to `--out_results`.

### 4. Inference

To evaluate a model or generate predictions on the test set:

**For Open-Source Models:**
```bash
python main.py \
    --mode infer \
    --config src/configs/train_blip2.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data/test.jsonl \
    --out_checkpoint ./outputs/blip2_checkpoints \
    --out_results ./outputs/blip2_results
```

**For API Models (e.g., GPT-4 Zero-Shot):**
```bash
python main.py \
    --mode infer \
    --config src/configs/train_gpt.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data/test.jsonl \
    --out_results ./outputs/gpt4_results \
    --api_key YOUR_API_KEY \
    --shots 0
```

**For API Models (e.g., GPT-4 One-Shot):**
By setting `--shots 1`, the script dynamically fetches an example from your training data to act as the few-shot prompt.
```bash
python main.py \
    --mode infer \
    --config src/configs/train_gpt.yaml \
    --image_dir ./data/images/COCO2017 \
    --jsonl_dir ./src/datasets/data/test.jsonl \
    --out_results ./outputs/gpt4_results \
    --api_key YOUR_API_KEY \
    --shots 1
```

### 5. Evaluation

If you only want to compute metrics (Accuracy, Precision, Recall, F1, and XYZ granular accuracy) from an existing prediction file:

```bash
python main.py \
    --mode eval \
    --out_results ./outputs/blip2_results
```
