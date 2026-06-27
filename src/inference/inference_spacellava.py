"""
Inference script for SpaceLLaVA.
Compatible with SpatialMQA evaluation protocol.

Uses the standalone llava library (install with: pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps)
SpaceLLaVA is NOT compatible with transformers.LlavaForConditionalGeneration
because it uses the original LLaVA format (based on llava-v1.5-13b).
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
from ..metrics.metrics import calculate_spatial_metrics
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
    disable_torch_init()

    model_path = config.model.model_name_or_path
    model_name = get_model_name_from_path(model_path)

    logger.info(f"Loading SpaceLLaVA model from {model_path}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None, model_name=model_name
    )

    # Load LoRA weights if available
    if hasattr(args, 'out_checkpoint') and args.out_checkpoint and Path(args.out_checkpoint).exists():
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

    # Detect conversation mode
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

    # Inference parameters matching spacellava_test.py reference
    temperature = 0.9
    top_p = None
    num_beams = 1
    max_new_tokens = 512

    predictions = []
    right_count = 0
    count = 0
    total = len(test_data)

    logger.info(f"Starting SpaceLLaVA inference on {total} samples...")
    logger.info(f"Parameters: temperature={temperature}, num_beams={num_beams}, max_new_tokens={max_new_tokens}")

    for index, item in enumerate(tqdm(test_data, desc="Inference", unit="img", ncols=100)):
        question = item["question"]
        options = item["options"]
        answer = item["answer"]
        image_name = item["image"]
        image_filepath = image_dir / image_name

        # SpaceLLaVA prompt format from spacellava_test.py reference (line 162)
        question_text = (
            f"Question: {question} \\n"
            f"Options: {'; '.join(options)} \\n"
            "Answer:"
        )

        # Handle IMAGE_PLACEHOLDER (same as reference lines 82-93)
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

        # Build conversation prompt
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process image
        image = load_image(image_filepath)
        images_tensor = process_images([image], image_processor, model.config)
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)

        # Tokenize
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        # Decode
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        output = outputs

        count += 1

        if len(output) == 0:
            output = "--"

        # Same matching logic as reference
        output_lower = output.lower()
        is_correct = 0
        if output_lower in answer.lower() or answer.lower() in output_lower:
            is_correct = 1
            right_count += 1

        predictions.append(build_result_record(item, index, output))

        # Log progress
        if (index + 1) % 20 == 0 or index == 0:
            acc = right_count / count
            logger.info(
                f"[{count}/{total}] Output: '{output_lower}' | Answer: '{answer}' | "
                f"Correct: {right_count}/{count} ({acc:.2%})"
            )

    # ── Save results ────────────────────────────────────────────────────
    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    # ── Calculate and log metrics ───────────────────────────────────────
    metrics = calculate_spatial_metrics(predictions)
    accuracy = right_count / count if count > 0 else 0.0
    logger.info("=" * 60)
    logger.info("--- SpaceLLaVA Evaluation Results ---")
    logger.info(f"Total samples: {count}")
    logger.info(f"Correct: {right_count}/{count}")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X (Left/Right):  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y (Above/Below): {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z (Front/Behind): {metrics['accuracy_z']:.4f}")
    logger.info(f"Results saved to: {out_path}")
    logger.info("=" * 60)

    return accuracy
