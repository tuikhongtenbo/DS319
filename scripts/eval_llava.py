import os
import argparse
import torch
import json
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from io import BytesIO
import re

# ======================================================================
# PATH CONFIGURATION
# ======================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_FILE_PATH = os.path.join(PROJECT_ROOT, 'outputs/llava_logs/test_results_llava.jsonl')
FILE_PATH = os.path.join(PROJECT_ROOT, 'src/datasets/data/dev.jsonl')
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data/images/COCO2017/')
PEFT_MODEL_ID = os.path.join(PROJECT_ROOT, 'outputs/llava_checkpoints/saved_model')
# ======================================================================

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

# Model
disable_torch_init()
model_path = 'liuhaotian/llava-v1.5-7b'
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "conv_mode": None,
    "sep": ",",
    "temperature": 0.4,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, None, model_name
)

if os.path.exists(PEFT_MODEL_ID):
    print(f"Loading LoRA weights from {PEFT_MODEL_ID}")
    model.load_adapter(PEFT_MODEL_ID)
else:
    print(f"[WARNING] Adapter path {PEFT_MODEL_ID} not found. Running with base model.")

def eval_model(args, question, image_file):
    qs = question
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        pass
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_files = [image_file]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

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
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

count = 0
right_count = 0

os.makedirs(os.path.dirname(RESULT_FILE_PATH), exist_ok=True)
with open(FILE_PATH, 'r', encoding="utf-8") as f, open(RESULT_FILE_PATH, 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data.get('id', count)
        options = data['options']
        image_name = data['image']
        image_filepath = os.path.join(IMAGE_DIR, image_name)
        
        qs = f'You are currently a senior expert in spatial relation reasoning. \n Given an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason. \n Input: Image: <image>, Question: {question}, Options: {"; ".join(options)}. \n Output:'
        output = eval_model(args, qs, image_filepath)
        
        count += 1
        if len(output) == 0:
            output = '--'
        if output.lower() in data['answer'].lower():
            result_json = {'id': id, 'result': 1, "output": output.lower(), "answer": data['answer']}
            fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            right_count += 1
        elif data['answer'].lower() in output.lower():
            result_json = {'id': id, 'result': 1, "output": output.lower(), "answer": data['answer']}
            fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            right_count += 1
        else:
            result_json = {'id': id, 'result': 0, "output": output.lower(), "answer": data['answer']}
            fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            
        print(f'[{count}] Output: {output.lower()} | Answer: {data["answer"].lower()} | Accuracy: {right_count}/{count} = {right_count / count:.4f}')

accuracy = right_count / count if count > 0 else 0.0
print(f'\n=================================')
print(f'Final LLaVA Accuracy: {accuracy:.4f}')
print(f'=================================')
