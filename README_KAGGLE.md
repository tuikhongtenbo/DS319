# SpatialMQA (Kaggle Edition)

This guide walks you through fine-tuning and running inference for the SpatialMQA repository on Kaggle using Kaggle Notebooks (with P100 or T4x2 GPUs).

### 🛠️ Decoupled Architecture
This codebase uses a fully decoupled structure where each model has its own dedicated training script in `src/trainers/` and inference script in `src/inference/`. This prevents cross-model dependency errors when running on Kaggle. The main dispatcher (`main.py`) dynamically imports the correct model logic based on your `--config` parameter, meaning your training and inference commands remain unified and unchanged.

## 🚀 Setting Up on Kaggle

1. **Create a New Notebook** on Kaggle and turn on the **GPU** (T4x2 or P100) under the "Accelerator" settings.
2. **Turn on Internet Connection** in the notebook settings.
3. **Upload the Codebase**: You can zip this repository and upload it as a Kaggle Dataset, then attach it to your notebook. Alternatively, `git clone` the repository directly in a cell.
4. **Prepare the Images**: You can either upload the `COCO2017` image folder as a separate Kaggle Dataset and attach it, OR download it dynamically using `wget` in a notebook cell as shown below.

### 1. Installation & Data Preparation

Install the requirements and download the COCO 2017 test dataset directly in a Notebook cell.

```python
# Unzip code if uploaded as dataset, or clone
!cp -r /kaggle/input/your-repo-dataset/DS319 /kaggle/working/
%cd /kaggle/working/DS319

# Install specific model requirements
!pip install -r src/requirements/requirement_blip.txt

# Download COCO 2017 test images
!mkdir -p data/images
%cd data/images
!wget http://images.cocodataset.org/zips/test2017.zip
!unzip -q test2017.zip
!mv test2017 COCO2017
!rm test2017.zip
%cd ../..
```

### 2. Fine-Tuning

Kaggle cell for training. Make sure to point the `--image_dir` and `--jsonl_dir` to your Kaggle mounted inputs.

```python
!python main.py \
    --mode train \
    --config src/configs/train_blip2.yaml \
    --image_dir /kaggle/input/coco2017-dataset/images \
    --jsonl_dir /kaggle/input/spatialmqa-dataset/src/datasets/data \
    --out_checkpoint /kaggle/working/outputs/blip2_checkpoints \
    --out_results /kaggle/working/outputs/blip2_results \
    --batch_size 8
```
*Note: Kaggle has a persistent `/kaggle/working/` directory. All checkpoints and results saved here can be downloaded after the notebook finishes.*

### 3. Inference

To run inference on Kaggle using a trained open-source model:

```python
!python main.py \
    --mode infer \
    --config src/configs/train_blip2.yaml \
    --image_dir /kaggle/input/coco2017-dataset/images \
    --jsonl_dir /kaggle/input/spatialmqa-dataset/src/datasets/data/test.jsonl \
    --out_checkpoint /kaggle/working/outputs/blip2_checkpoints \
    --out_results /kaggle/working/outputs/blip2_results
```

#### API Models (GPT-4 / Qwen) on Kaggle

If you are evaluating API models, **DO NOT hardcode your API Key in your code**. Use Kaggle Secrets.

1. Go to **"Add-ons" -> "Secrets"** in your notebook menu.
2. Add a new secret with Label: `OPENAI_API_KEY` and Value: `<your-api-key>`.
3. Read it securely in the notebook cell:

```python
from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("OPENAI_API_KEY")

# Use os.system to avoid API keys leaking into output logs
os.system(f"""
python main.py \
    --mode infer \
    --config src/configs/train_gpt.yaml \
    --image_dir /kaggle/input/coco2017-dataset/images \
    --jsonl_dir /kaggle/input/spatialmqa-dataset/src/datasets/data/test.jsonl \
    --out_results /kaggle/working/outputs/gpt4_results \
    --api_key {api_key} \
    --shots 1
""")
```

### 4. Downloading Results

Once your inference or training is done, you can zip the output directory to easily download it from Kaggle's output pane:

```python
!zip -r /kaggle/working/results.zip /kaggle/working/outputs/
```
