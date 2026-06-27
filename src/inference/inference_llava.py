"""
HuggingFace Transformers-native inference for LLaVA.

Uses transformers.LlavaForConditionalGeneration + AutoProcessor
(NOT the standalone llava library which requires old torch==2.1.2).

Same prompt format & generation parameters as spatial_test_llava.py reference.
"""

import re
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration

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


def _build_prompt(question_text: str) -> str:
    """
    Wrap question text into LLaVA-v1.5 chat format.

    This replicates what conv_templates["llava_v1"] produces:
        USER: <image>\n{question}\nASSISTANT:

    Since _build_question already embeds <image> inside the text,
    we just wrap it with the USER/ASSISTANT format.
    """
    return f"USER: {question_text}\nASSISTANT:"


def run_infer(args, config: ExperimentConfig):
    """
    Main inference loop using HuggingFace Transformers LLaVA.
    """
    model_path = config.model.model_name_or_path

    logger.info(f"Loading LLaVA model from {model_path} via HuggingFace Transformers...")

    # ── Load model and processor ────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
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

    # ── Load dataset ────────────────────────────────────────────────────
    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    # ── Inference parameters (matching reference) ───────────────────────
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

        # Build full prompt with chat format
        prompt = _build_prompt(question_text)

        # Load and process image
        try:
            image = load_image(image_filepath)
        except Exception as e:
            logger.warning(f"Failed to load image {image_filepath}: {e}")
            predictions.append(build_result_record(item, index, "--"))
            count += 1
            continue

        # Process inputs through the HF processor
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        # Decode — strip the input prompt from the output
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

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
