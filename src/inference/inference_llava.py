"""
Inference script for LLaVA using vLLM client.
Compatible with SpatialMQA evaluation protocol.
"""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import build_result_record, resolve_test_path
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for API request."""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format=img.format or "JPEG")
        return base64.b64encode(buffered.getvalue()).decode()


def build_spatial_prompt(question: str, options: list) -> str:
    """Build prompt for spatial reasoning task matching SpatialMQA format."""
    options_str = "; ".join(options)
    return (
        f"You are a spatial reasoning expert. Given an image, a question, and options, "
        f"choose the correct spatial relation. Answer with ONLY the option (no explanation).\n"
        f"Question: {question}\n"
        f"Options: {options_str}\n"
        f"Answer:"
    )


def run_infer(args, config: ExperimentConfig):
    from vllm import LLM, SamplingParams
    from openai import OpenAI

    vllm_host = args.vllm_host or "http://localhost:8000"
    logger.info(f"Connecting to vLLM server at {vllm_host}")

    # Connect to vLLM server
    client = OpenAI(api_key="EMPTY", base_url=f"{vllm_host}/v1")

    # Check server health
    try:
        client.models.list()
        logger.info("Connected to vLLM server successfully")
    except Exception as e:
        logger.error(f"Failed to connect to vLLM server: {e}")
        raise

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    right_count = 0
    total = len(test_data)

    # Inference with progress bar showing numbers
    logger.info("Starting LLaVA inference via vLLM...")
    for index, item in enumerate(tqdm(test_data, desc="Inference", unit="img")):
        question = item["question"]
        options = item["options"]
        answer = item["answer"]
        image_name = item["image"]
        image_filepath = image_dir / image_name

        # Build prompt
        prompt_text = build_spatial_prompt(question, options)

        # Encode image
        try:
            image_base64 = encode_image_to_base64(str(image_filepath))
        except Exception as e:
            logger.warning(f"Failed to load image {image_filepath}: {e}")
            output = "--"
            predictions.append(build_result_record(item, index, output))
            continue

        # Prepare message for chat API
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        # Call vLLM API
        try:
            response = client.chat.completions.create(
                model=config.model.model_name_or_path,
                messages=messages,
                max_tokens=20,
                temperature=0.0,
                stop=["\n", ".", ";"],
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API error at index {index}: {e}")
            output = "--"

        # Post-process output
        if not output or len(output) == 0:
            output = "--"

        # Clean output - extract first word/phrase that matches an option
        output_clean = output.lower().split()[0] if output != "--" else output
        for opt in options:
            if opt.lower() in output_clean:
                output = opt
                break

        # Check correctness
        is_correct = 0
        output_lower = output.lower()
        answer_lower = answer.lower()
        if output_lower == answer_lower or answer_lower in output_lower:
            is_correct = 1
            right_count += 1

        predictions.append(build_result_record(item, index, output))

        # Log progress every 10 items
        if (index + 1) % 10 == 0 or index == 0:
            acc = right_count / (index + 1)
            logger.info(
                f"[{index + 1}/{total}] Output: '{output}' | Answer: '{answer}' | "
                f"Correct: {right_count}/{index + 1} ({acc:.2%})"
            )

    # Save results
    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    # Final metrics
    accuracy = right_count / total if total > 0 else 0.0
    logger.info("=" * 50)
    logger.info("--- LLaVA (vLLM) Final Results ---")
    logger.info(f"Total samples: {total}")
    logger.info(f"Correct: {right_count}/{total}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    logger.info(f"Results saved to: {out_path}")
    logger.info("=" * 50)

    return accuracy
