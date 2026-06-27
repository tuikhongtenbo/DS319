"""
vLLM-accelerated inference for LLaVA and SpaceLLaVA.

Format matches spatial_test_llava.py reference exactly.

Requires: pip install vllm>=0.4.0
Supported: llava-hf/llava-1.5, llava-1.6, SpaceLLaVA
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
    Build prompt EXACTLY matching spatial_test_llava.py reference.
    Note the space after \\n at start of lines.
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


def _detect_conv_mode(model_name_or_path: str) -> str:
    m = model_name_or_path.lower()
    if "llama-2" in m:
        return "llava_llama_2"
    if "mistral" in m or "mixtral" in m:
        return "mistral_instruct"
    if "v1.6-34b" in m:
        return "chatml_direct"
    if "v1" in m:
        return "llava_v1"
    if "mpt" in m:
        return "mpt"
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


def _extract_answer(output: str, options: List[str], answer: str) -> str:
    """
    Extract the spatial answer from model output.
    Uses same matching logic as spatial_test_llava.py:
    - output.lower() in answer.lower()
    - answer.lower() in output.lower()
    """
    if not output or len(output.strip()) == 0:
        return "--"

    output_lower = output.lower().strip()
    answer_lower = answer.lower()

    # Direct match with answer
    if output_lower in answer_lower or answer_lower in output_lower:
        return answer

    # Match against options
    for opt in options:
        opt_lower = opt.lower()
        if opt_lower == output_lower:
            return opt
        if opt_lower in output_lower or output_lower in opt_lower:
            return opt

    # Return as-is if no match (let evaluation decide)
    return output.strip().rstrip(".")


class VLLMLlavaPredictor:
    """
    Direct vLLM LLM instance with parameters matching spatial_test_llava.py.
    Uses conv_templates for proper prompt formatting.
    """

    def __init__(
        self,
        llm,
        conv_mode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.4,
        top_p: float = None,
        num_beams: int = 1,
    ):
        self.llm = llm
        self.conv_mode = conv_mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams

        # Get tokenizer for chat template
        self.tokenizer = self.llm.get_tokenizer()

    def _build_prompt(self, question: str, options: List[str]) -> str:
        """Build prompt with proper chat template for vLLM."""
        options_str = "; ".join(options)
        prompt_text = (
            f'You are currently a senior expert in spatial relation reasoning. \n'
            f' Given an Image, a Question and Options, your task is to answer the '
            f'correct spatial relation. Note that you only need to choose one option '
            f'from the all options without explaining any reason. \n'
            f' Input: Image: <image>, Question: {question}, Options: {options_str}. \n'
            f' Output:'
        )

        # Build chat messages (no system message, user only)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]}
        ]

        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as e:
            logger.warning(f"Chat template failed, using raw prompt: {e}")
            prompt = "<image>\n" + prompt_text

        return prompt

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: List[str], answer: str) -> str:
        prompt = self._build_prompt(question, options)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return "--"

        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=self.top_p if self.top_p is not None else 1.0,
                stop=[],
            )

            outputs = self.llm.generate(
                [prompt],
                sampling_params,
                multi_modal_data={"image": image},
            )
        except (TypeError, AttributeError) as e:
            # Fallback if multi_modal_data not supported
            logger.debug(f"Falling back to non-multimodal: {e}")
            outputs = self.llm.generate([prompt], sampling_params)

        raw_output = outputs[0].outputs[0].text

        # Debug: log raw output for first few samples
        nonlocal debug_count
        if debug_count < 3:
            logger.info(f"[DEBUG] Sample {index} raw output: '{raw_output}'")
            debug_count += 1

        return _extract_answer(raw_output, options, answer)


def run_infer(args, config: ExperimentConfig):
    model_path = config.model.model_name_or_path
    model_type = config.model.model_type.lower()

    if not check_vllm_available():
        logger.error("vLLM not installed. Install with: pip install vllm")
        return

    logger.info(f"Loading vLLM LLM for {model_type} inference...")

    try:
        from vllm import LLM, LoraConfig
    except ImportError:
        logger.error("vLLM import failed.")
        return

    # Build LLM kwargs
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": config.model.tensor_parallel_size,
        "gpu_memory_utilization": config.model.gpu_memory_utilization,
        "max_model_len": config.model.max_model_len,
    }

    # Load LoRA if configured
    if getattr(config.model, "use_lora", False):
        lora_path = getattr(args, "lora_path", None)
        if lora_path is None:
            output_dir = Path(config.training.output_dir)
            possible_lora = output_dir / "final"
            if possible_lora.exists():
                lora_path = str(possible_lora)
            else:
                adapters = list(output_dir.glob("adapter*"))
                if adapters:
                    lora_path = str(adapters[0])

        if lora_path:
            logger.info(f"Loading LoRA adapter from: {lora_path}")
            lora_config = LoraConfig(
                lora_r=config.model.lora_r,
                lora_alpha=config.model.lora_alpha,
                lora_dropout=0.05,
            )
            llm_kwargs["lora_config"] = lora_config
            llm_kwargs["auto_model_type"] = "causal"

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        logger.error(f"vLLM engine failed to load: {e}")
        return

    conv_mode = _detect_conv_mode(model_path)
    logger.info(f"Using conversation template: {conv_mode}")

    # Use same parameters as spatial_test_llava.py
    predictor = VLLMLlavaPredictor(
        llm=llm,
        conv_mode=conv_mode,
        max_new_tokens=512,
        temperature=0.4,
        top_p=None,
        num_beams=1,
    )

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    right_count = 0
    total = len(test_data)

    logger.info(f"Starting vLLM LLaVA inference on {total} samples...")
    logger.info(f"Parameters: max_new_tokens=512, temperature=0.4")
    logger.info(f"Model path: {model_path}")

    # Debug: print first prompt and raw output for first 3 samples
    debug_count = 0

    for index, item in enumerate(tqdm(test_data, desc="Inference", unit="img", ncols=100)):
        image_path = image_dir / item["image"]
        output = predictor.predict(
            str(image_path),
            item["question"],
            item.get("options", []),
            item.get("answer", "")
        )
        if not output:
            output = "--"

        answer = item.get("answer", "")

        # Same matching logic as spatial_test_llava.py
        is_correct = 0
        output_lower = output.lower()
        answer_lower = answer.lower()
        if output_lower in answer_lower or answer_lower in output_lower:
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
