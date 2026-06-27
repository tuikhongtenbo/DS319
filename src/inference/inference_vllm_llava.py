"""
vLLM-accelerated inference for LLaVA and SpaceLLaVA.

Supports two modes:
1. vLLM Server mode (--vllm_host): Connect to running vLLM server via OpenAI-compatible API
2. Direct LLM mode: Create vLLM LLM instance directly

Requires: pip install vllm>=0.4.0
Supported: llava-hf/llava-1.5, llava-1.6, SpaceLLaVA
Qwen2-VL: use inference_vllm_qwen.py instead.
"""

import base64
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import build_result_record, resolve_test_path
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger
from .inference_vllm_base import check_vllm_available, build_sampling_params

logger = setup_logger(__name__)

IMAGE_PLACEHOLDER = "<image>"


def build_spatial_prompt(question: str, options: List[str]) -> str:
    """
    Build prompt matching SpatialMQA reference format exactly.
    This format has been verified to work with llava-1.5-7b.
    """
    options_str = "; ".join(options)
    return (
        "You are currently a senior expert in spatial relation reasoning. \n"
        "Given an Image, a Question and Options, your task is to answer the "
        "correct spatial relation. Note that you only need to choose one option "
        "from the all options without explaining any reason. \n"
        f"Input: Image: <image>, Question: {question}, Options: {options_str}. \n"
        "Output:"
    )


def _detect_conv_mode(model_name_or_path: str) -> str:
    m = model_name_or_path.lower()
    if "llama-2" in m:
        return "llava_llama_2"
    if "mistral" in m or "mixtral" in m:
        return "mistral_instruct"
    if "v1" in m:
        return "llava_v1"
    return "llava_v0"


def _encode_image(image_path: str) -> Optional[str]:
    """Encode image to base64 string."""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format=img.format or "JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def _extract_answer(output: str, options: List[str]) -> str:
    """
    Extract the spatial answer from model output.
    Handles various formats the model might produce.
    """
    if not output or len(output.strip()) == 0:
        return "--"

    output_lower = output.lower().strip()

    # Direct match
    for opt in options:
        opt_lower = opt.lower()
        if opt_lower == output_lower:
            return opt

    # Check if output contains an option
    for opt in options:
        opt_lower = opt.lower()
        # Match whole words only
        pattern = r'\b' + re.escape(opt_lower) + r'\b'
        if re.search(pattern, output_lower):
            return opt

    # If output is too short/long or looks like garbage, try to extract first word
    words = output_lower.split()
    if len(words) <= 3:
        for opt in options:
            if any(opt.lower() in word for word in words):
                return opt

    # Return cleaned output if it doesn't match any option
    return output.strip().rstrip(".")


class VLLMAPIPredictor:
    """Connect to running vLLM server via OpenAI-compatible API."""

    def __init__(self, vllm_host: str, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(api_key="EMPTY", base_url=f"{vllm_host}/v1")
        self.model_name = model_name
        # Verify connection
        try:
            self.client.models.list()
            logger.info(f"Connected to vLLM server at {vllm_host}")
        except Exception as e:
            logger.error(f"Failed to connect to vLLM server: {e}")
            raise

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        prompt_text = build_spatial_prompt(question, options)

        # Encode image to base64
        image_b64 = _encode_image(image_path)
        if image_b64 is None:
            return "--"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=30,  # Increased slightly
                temperature=0.0,
                stop=[],  # Let model decide when to stop
            )
            output = response.choices[0].message.content
        except Exception as e:
            logger.warning(f"API error: {e}")
            return "--"

        return _extract_answer(output, options)


class VLLMLlavaPredictor:
    """Direct vLLM LLM instance (original mode)."""

    def __init__(self, llm, conv_mode: str, max_new_tokens: int = 30):
        self.llm = llm
        self.conv_mode = conv_mode
        # No stop tokens - let model generate naturally
        self.sampling_params = build_sampling_params(max_new_tokens, temperature=0.0)
        self.sampling_params.stop = []

    def _build_prompt(self, question: str, options: List[str]) -> str:
        """Build prompt with image token at start."""
        prompt_text = build_spatial_prompt(question, options)
        # Ensure <image> is at the start
        if IMAGE_PLACEHOLDER not in prompt_text:
            prompt_text = IMAGE_PLACEHOLDER + "\n" + prompt_text
        return prompt_text

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        prompt = self._build_prompt(question, options)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return "--"

        try:
            outputs = self.llm.generate(
                [prompt],
                self.sampling_params,
                multi_modal_data={"image": image},
            )
        except (TypeError, AttributeError):
            # Fallback if multi_modal_data not supported
            outputs = self.llm.generate([prompt], self.sampling_params)

        raw_output = outputs[0].outputs[0].text
        return _extract_answer(raw_output, options)


def run_infer(args, config: ExperimentConfig):
    model_path = config.model.model_name_or_path
    model_type = config.model.model_type.lower()

    # Check if connecting to vLLM server (API mode) or using direct LLM
    vllm_host = getattr(args, "vllm_host", None)

    if vllm_host:
        # API Client Mode - connect to running vLLM server
        logger.info(f"Using vLLM API client mode, connecting to {vllm_host}")
        predictor = VLLMAPIPredictor(vllm_host=vllm_host, model_name=model_path)
    else:
        # Direct LLM Mode - create vLLM instance
        if not check_vllm_available():
            logger.error("vLLM not installed. Install with: pip install vllm")
            return

        logger.info(f"Loading vLLM LLM for {model_type} inference...")

        try:
            from vllm import LLM
        except ImportError:
            logger.error("vLLM import failed.")
            return

        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=config.model.tensor_parallel_size,
                gpu_memory_utilization=config.model.gpu_memory_utilization,
                max_model_len=config.model.max_model_len,
            )
        except Exception as e:
            logger.error(f"vLLM engine failed to load: {e}")
            return

        conv_mode = _detect_conv_mode(model_path)
        logger.info(f"Using conversation template: {conv_mode}")
        predictor = VLLMLlavaPredictor(llm=llm, conv_mode=conv_mode, max_new_tokens=30)

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    right_count = 0
    total = len(test_data)

    logger.info(f"Starting vLLM LLaVA inference on {total} samples...")

    # Use enumerate(tqdm(...)) to show item numbers in progress bar
    for index, item in enumerate(tqdm(test_data, desc="Inference", unit="img", ncols=100)):
        image_path = image_dir / item["image"]
        output = predictor.predict(str(image_path), item["question"], item.get("options", []))
        if not output:
            output = "--"

        answer = item.get("answer", "")
        is_correct = 0
        if output.lower() == answer.lower() or answer.lower() in output.lower():
            is_correct = 1
            right_count += 1

        predictions.append(build_result_record(item, index, output))

        # Log progress every 20 items
        if (index + 1) % 20 == 0 or index == 0:
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

    # Calculate metrics
    metrics = calculate_spatial_metrics(predictions)
    logger.info("=" * 60)
    logger.info("--- vLLM LLaVA Evaluation Results ---")
    logger.info(f"Total samples: {total}")
    logger.info(f"Correct: {right_count}/{total}")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X (Left/Right):  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y (Above/Below): {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z (Front/Behind): {metrics['accuracy_z']:.4f}")
    logger.info(f"Results saved to: {out_path}")
    logger.info("=" * 60)
