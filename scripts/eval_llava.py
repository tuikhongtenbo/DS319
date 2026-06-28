"""
LLaVA Evaluation Script with LoRA fine-tuned weights.

Based on inference_llava.py patterns with custom eval logic.
LoRA adapter: outputs/llava_checkpoints/saved_model/checkpoint-2000
"""

import os
import argparse
import re
import json
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

# ======================================================================
# PATH CONFIGURATION
# ======================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_FILE_PATH = os.path.join(PROJECT_ROOT, 'outputs/llava_logs/test_results_llava.jsonl')
FILE_PATH = os.path.join(PROJECT_ROOT, 'src/datasets/data/test.jsonl')
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data/images/COCO2017/')
PEFT_MODEL_ID = os.path.join(PROJECT_ROOT, 'outputs/llava_checkpoints/saved_model/checkpoint-2000')
# ======================================================================


def load_image(image_file):
    if str(image_file).startswith("http") or str(image_file).startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(str(image_file)).convert("RGB")
    return image


def _build_question(question: str, options: list) -> str:
    """Build the prompt text matching training format."""
    options_str = "; ".join(options)
    return (
        f'You are currently a senior expert in spatial relation reasoning. \\n'
        f' Given an Image, a Question and Options, your task is to answer the '
        f'correct spatial relation. Note that you only need to choose one option '
        f'from the all options without explaining any reason. \\n'
        f' Input: Image: <image>, Question: {question}, Options: {options_str}. \\n'
        f' Output:'
    )


# ======================================================================
# Model Loading
# ======================================================================
disable_torch_init()
model_path = 'liuhaotian/llava-v1.5-7b'
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)
model.to(torch.bfloat16)

# Load LoRA adapter
ckpt_path = Path(PEFT_MODEL_ID)
if ckpt_path.exists():
    if not (ckpt_path / "adapter_config.json").exists():
        # Try saved_model or checkpoint-* subdirs
        candidates = [
            ckpt_path / "saved_model",
            ckpt_path,
        ]
        candidates.extend(sorted(ckpt_path.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True))
        for candidate in candidates:
            if candidate.exists() and (candidate / "adapter_config.json").exists():
                peft_model_id = str(candidate)
                break
        else:
            peft_model_id = str(ckpt_path)
    else:
        peft_model_id = str(ckpt_path)
    
    print(f"Loading LoRA weights from {peft_model_id}")
    model.load_adapter(peft_model_id)
else:
    print(f"[INFO] Adapter path {PEFT_MODEL_ID} not found. Running with BASE model.")

model.eval()


# ======================================================================
# Inference Parameters (matching inference_llava.py)
# ======================================================================
TEMPERATURE = 0.0  # greedy for deterministic output
TOP_P = None
NUM_BEAMS = 1
MAX_NEW_TOKENS = 32  # was 8, longer for multi-word answers

# Conv mode inference
def get_conv_mode():
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    elif "mistral" in model_name.lower():
        return "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        return "chatml_direct"
    elif "v1" in model_name.lower():
        return "llava_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "llava_v0"

CONV_MODE = get_conv_mode()
print(f"Using conv_mode: {CONV_MODE}")


def eval_model(question: str, image_file: str) -> str:
    """Evaluate single sample."""
    # Build question text
    question_text = question
    
    # Process image token
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in question_text:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, question_text)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question_text)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + question_text
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question_text

    # Build conversation
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Load and process image
    try:
        image = load_image(image_file)
    except Exception as e:
        print(f"Warning: Failed to load image {image_file}: {e}")
        return "--"
    
    image_sizes = [image.size]
    images_tensor = process_images(
        [image],
        image_processor,
        model.config,
    ).to(model.device, dtype=torch.bfloat16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if TEMPERATURE > 0 else False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=NUM_BEAMS,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output


# ======================================================================
# Answer Normalization
# ======================================================================
SPATIAL_MAPPING = {
    'left of': ['left', 'left of', 'left-of'],
    'right of': ['right', 'right of', 'right-of'],
    'on/above': ['above', 'on/above', 'above/on', 'on top of'],
    'below': ['below', 'under'],
    'behind': ['behind', 'in back of'],
    'in front of': ['in front of', 'front of', 'front'],
}


def normalize_answer(answer: str) -> str:
    """Normalize answer to standard format."""
    answer = answer.lower().strip()
    for key, variants in SPATIAL_MAPPING.items():
        if answer in variants or any(v in answer for v in variants):
            return key
    return answer


def extract_spatial_keyword(output: str) -> str:
    """Extract first spatial keyword from output."""
    output_lower = output.lower().strip()
    for key, variants in SPATIAL_MAPPING.items():
        if any(v in output_lower for v in variants):
            return key
    return output_lower.split()[0] if output_lower else output_lower


# ======================================================================
# Evaluation
# ======================================================================
print(f"\n{'='*60}")
print(f"LLaVA Evaluation")
print(f"Model: {model_path}")
print(f"LoRA: {PEFT_MODEL_ID if os.path.exists(PEFT_MODEL_ID) else 'BASE MODEL'}")
print(f"{'='*60}\n")

os.makedirs(os.path.dirname(RESULT_FILE_PATH), exist_ok=True)

count = 0
right_count = 0
output_distribution = Counter()

with open(FILE_PATH, 'r', encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

with open(RESULT_FILE_PATH, 'w+', encoding="utf-8") as fout:
    for item in tqdm(test_data, desc="Evaluating", unit="img", ncols=100):
        question = item['question']
        options = item['options']
        answer = item['answer']
        image_name = item['image']
        image_filepath = os.path.join(IMAGE_DIR, image_name)
        sample_id = item.get('id', count)

        # Run inference
        output = eval_model(question, image_filepath)
        
        # Extract and normalize
        output_normalized = extract_spatial_keyword(output)
        answer_normalized = normalize_answer(answer)
        
        if not output_normalized:
            output_normalized = '--'

        count += 1
        output_distribution[output_normalized] += 1

        # Check correctness
        is_correct = output_normalized == answer_normalized
        if is_correct:
            right_count += 1

        # Write result
        result_json = {
            'id': sample_id,
            'result': 1 if is_correct else 0,
            'output': output_normalized,
            'answer': answer
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')

        # Progress log every 50 samples
        if count % 50 == 0:
            acc = right_count / count
            print(f"[{count}] Acc: {right_count}/{count} = {acc:.4f}")

# ======================================================================
# Final Results
# ======================================================================
accuracy = right_count / count if count > 0 else 0.0

print(f"\n{'='*60}")
print(f"Final Results")
print(f"{'='*60}")
print(f"Total samples: {count}")
print(f"Correct: {right_count}/{count}")
print(f"Accuracy: {accuracy:.4f}")
print(f"\nOutput distribution:")
for label, cnt in output_distribution.most_common():
    pct = cnt / count * 100
    print(f"  {label}: {cnt} ({pct:.1f}%)")
print(f"{'='*60}")
