"""
Inference for LLaVA with LoRA fine-tuned weights.

Uses the native llava library (load_pretrained_model, conv_templates,
process_images, tokenizer_image_token) - matching the reference
spatial_test_llava_lora.py from SpatialMQA exactly.
"""

import re
from pathlib import Path
from typing import List

import torch
from PIL import Image
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

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import build_result_record, resolve_test_path
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def load_image(image_file: str) -> Image.Image:
    """Load image from file path or URL."""
    if str(image_file).startswith("http") or str(image_file).startswith("https"):
        import requests
        from io import BytesIO
        response = requests.get(str(image_file))
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(str(image_file)).convert("RGB")


def _build_question(question: str, options: list) -> str:
    """
    Build the prompt text matching spatial_test_llava_lora.py line 165.

    Uses literal \\n inside the f-string (producing the two-character
    sequence backslash-n in the final string), matching the reference.
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


def run_infer(args, config: ExperimentConfig):
    """
    Main inference loop using native llava library.
    Matches the reference spatial_test_llava_lora.py.
    """
    model_path = config.model.model_name_or_path

    logger.info(f"Loading LLaVA model from {model_path} via llava.model.builder...")

    # ── Load model (matching reference lines 59-78) ─────────────────────
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    # Load LoRA weights if checkpoint path provided
    if hasattr(args, 'out_checkpoint') and args.out_checkpoint:
        peft_model_id = args.out_checkpoint
        # If the checkpoint dir itself doesn't have adapter_config.json,
        # look for subdirectories
        ckpt_path = Path(peft_model_id)
        if ckpt_path.exists():
            if not (ckpt_path / "adapter_config.json").exists():
                # Try saved_model or checkpoint-* subdirs
                candidates = [
                    ckpt_path / "saved_model",
                    ckpt_path,
                ]
                # Also check for checkpoint-* dirs
                candidates.extend(sorted(ckpt_path.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True))
                for candidate in candidates:
                    if candidate.exists() and (candidate / "adapter_config.json").exists():
                        peft_model_id = str(candidate)
                        break

            logger.info(f"Loading LoRA adapter from {peft_model_id}")
            model.load_adapter(peft_model_id)

    model.eval()

    # ── Infer conv_mode (matching reference lines 95-106) ───────────────
    conv_mode = None
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

    logger.info(f"Using conv_mode: {conv_mode}")

    # ── Load dataset ────────────────────────────────────────────────────
    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    # ── Inference parameters (matching reference line 67-70) ────────────
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

        # Build question text (matching reference line 165)
        question_text = _build_question(question, options)

        # ── Build prompt via conv_templates (matching reference lines 82-120) ──
        qs = question_text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # Only add image token at the beginning if <image> placeholder is not already in the text
        if IMAGE_PLACEHOLDER not in qs:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Load and process image
        try:
            image = load_image(image_filepath)
        except Exception as e:
            logger.warning(f"Failed to load image {image_filepath}: {e}")
            predictions.append(build_result_record(item, index, "--"))
            count += 1
            continue

        image_sizes = [image.size]
        images_tensor = process_images(
            [image],
            image_processor,
            model.config,
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # Generate (matching reference lines 136-147)
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

        # Decode (matching reference line 149)
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        count += 1

        if len(output) == 0:
            output = "--"

        # Same matching logic as reference (lines 170-180)
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
    logger.info("--- LLaVA LoRA Evaluation Results ---")
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
