"""
Inference script for SpaceLLaVA.
Compatible with SpatialMQA evaluation protocol.
"""

import os
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import build_result_record, resolve_test_path
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def load_image(image_file):
    if str(image_file).startswith("http") or str(image_file).startswith("https"):
        import requests
        from io import BytesIO
        response = requests.get(str(image_file))
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(str(image_file)).convert("RGB")


def run_infer(args, config: ExperimentConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id or config.model.device_map)

    disable_torch_init()

    model_path = config.model.model_name_or_path
    model_name = get_model_name_from_path(model_path)

    logger.info(f"Loading SpaceLLaVA model from {model_path}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None, model_name=model_name
    )

    if args.out_checkpoint and Path(args.out_checkpoint).exists():
        lora_path = Path(args.out_checkpoint) / "best_model"
        if not lora_path.exists():
            lora_path = Path(args.out_checkpoint) / "saved_model"
        if lora_path.exists():
            logger.info(f"Loading LoRA weights from {lora_path}")
            model.load_adapter(str(lora_path))

    model.eval()

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

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

    logger.info(f"Using conversation mode: {conv_mode}")

    predictions = []
    right_count = 0
    count = 0

    for index, item in enumerate(tqdm(test_data)):
        question = item["question"]
        options = item["options"]
        answer = item["answer"]
        image_name = item["image"]
        image_filepath = image_dir / image_name

        # SpaceLLaVA prompt format from SpatialMQA
        question_text = (
            f"Question: {question} \n"
            f"Options: {'; '.join(options)} \n"
            "Answer:"
        )

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in question_text:
            if model.config.mm_use_im_start_end:
                question_text = re.sub(IMAGE_PLACEHOLDER, image_token_se, question_text)
            else:
                question_text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question_text)
        elif model.config.mm_use_im_start_end:
            question_text = image_token_se + "\n" + question_text
        else:
            question_text = DEFAULT_IMAGE_TOKEN + "\n" + question_text

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = load_image(image_filepath)
        images_tensor = process_images([image], image_processor, model.config)
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=20,
                use_cache=True,
            )

        input_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if len(output) == 0:
            output = "--"

        output_lower = output.lower()
        is_correct = 0
        if output_lower in answer.lower() or answer.lower() in output_lower:
            is_correct = 1
            right_count += 1

        predictions.append(build_result_record(item, index, output))
        count += 1

        logger.info(f"Output: {output_lower} | Answer: {answer} | Correct: {right_count}/{count}")

    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    accuracy = right_count / count if count > 0 else 0.0
    logger.info(f"--- SpaceLLaVA Final Results ---")
    logger.info(f"Accuracy: {accuracy:.4f} ({right_count}/{count})")
