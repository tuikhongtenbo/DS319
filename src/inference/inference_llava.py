"""
HuggingFace-native inference for LLaVA.

Follows spatial_test_llava.py reference exactly:
- Uses llava library: load_pretrained_model, conv_templates, tokenizer_image_token
- Same prompt format, same generation parameters (temperature=0.4, num_beams=1, max_new_tokens=512)
- Same matching logic for evaluation
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
    """Load image from file path or URL."""
    if str(image_file).startswith("http") or str(image_file).startswith("https"):
        import requests
        from io import BytesIO
        response = requests.get(str(image_file))
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(str(image_file)).convert("RGB")


def _detect_conv_mode(model_name: str) -> str:
    """Auto-detect conversation mode from model name, matching the reference."""
    m = model_name.lower()
    if "llama-2" in m:
        return "llava_llama_2"
    if "mistral" in m:
        return "mistral_instruct"
    if "v1.6-34b" in m:
        return "chatml_direct"
    if "v1" in m:
        return "llava_v1"
    if "mpt" in m:
        return "mpt"
    return "llava_v0"


def _build_question(question: str, options: list) -> str:
    """
    Build the prompt text EXACTLY matching spatial_test_llava.py line 163.

    The reference uses literal \\n inside the f-string (producing the
    two-character sequence backslash-n in the final string).
    """
    options_str = "; ".join(options)
    return (
        f'You are currently a senior expert in spatial relation reasoning. \\n'
        f' Given an Image, a Question and Options, your task is to answer the '
        f'correct spatial relation. Note that you only need to choose one option '
        f'from the all options without explaining any reason. \\n'
        f' Input: Image: <image>, Question: {question}, Options: {options_str}. \\n'
        f' Output:'
    )


def eval_model(
    model, tokenizer, image_processor,
    question_text: str, image_file: str, conv_mode: str,
    temperature: float = 0.4, top_p=None, num_beams: int = 1,
    max_new_tokens: int = 512,
) -> str:
    """
    Run inference on a single sample, following spatial_test_llava.py eval_model() exactly.

    Steps:
    1. Handle IMAGE_PLACEHOLDER / mm_use_im_start_end
    2. Build conversation via conv_templates
    3. Process image through image_processor
    4. Tokenize with tokenizer_image_token
    5. Generate with model.generate()
    6. Decode output
    """
    qs = question_text

    # ── Step 1: Insert image token into the prompt ──────────────────────
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

    # ── Step 2: Build conversation prompt ───────────────────────────────
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # ── Step 3: Process image ───────────────────────────────────────────
    image = load_image(image_file)
    image_sizes = [image.size]
    images_tensor = process_images(
        [image], image_processor, model.config
    ).to(model.device, dtype=torch.float16)

    # ── Step 4: Tokenize ────────────────────────────────────────────────
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # ── Step 5: Generate ────────────────────────────────────────────────
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    # ── Step 6: Decode ──────────────────────────────────────────────────
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_infer(args, config: ExperimentConfig):
    """
    Main inference loop matching spatial_test_llava.py structure.
    """
    model_path = config.model.model_name_or_path

    # ── Load model (same as reference lines 59-76) ──────────────────────
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    logger.info(f"Loading LLaVA model from {model_path} (name: {model_name})...")

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

    # ── Detect conversation mode ────────────────────────────────────────
    conv_mode = _detect_conv_mode(model_name)
    logger.info(f"Using conversation mode: {conv_mode}")

    # ── Load dataset ────────────────────────────────────────────────────
    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    # ── Inference parameters (matching reference lines 61-71) ───────────
    temperature = 0.4
    top_p = None
    num_beams = 1
    max_new_tokens = 512

    predictions = []
    right_count = 0
    count = 0
    total = len(test_data)

    logger.info(f"Starting LLaVA inference on {total} samples...")
    logger.info(f"Parameters: temperature={temperature}, num_beams={num_beams}, max_new_tokens={max_new_tokens}")

    for index, item in enumerate(tqdm(test_data, desc="Inference", unit="img", ncols=100)):
        question = item["question"]
        options = item["options"]
        answer = item["answer"]
        image_name = item["image"]
        image_filepath = str(image_dir / image_name)

        # Build question text (same as reference line 163)
        question_text = _build_question(question, options)

        # Run model inference
        output = eval_model(
            model, tokenizer, image_processor,
            question_text, image_filepath, conv_mode,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

        count += 1

        if len(output) == 0:
            output = "--"

        # Same matching logic as reference (lines 168-178)
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
    logger.info("--- LLaVA Evaluation Results ---")
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
