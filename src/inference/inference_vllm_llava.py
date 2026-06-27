"""
vLLM-accelerated inference for LLaVA and SpaceLLaVA.

Format matches spatial_test_llava.py reference exactly.

Requires: pip install vllm>=0.4.0
Supported: llava-hf/llava-1.5, llava-1.6, SpaceLLaVA
"""

import re
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
from .inference_vllm_base import check_vllm_available

logger = setup_logger(__name__)


def build_spatial_prompt(question: str, options: List[str]) -> str:
    """
    Build prompt EXACTLY matching spatial_test_llava.py reference.

    The reference uses literal \\n inside the f-string (which produces the
    two-character sequence backslash-n in the final string). We replicate
    that here so the model sees the identical prompt.
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


class VLLMLlavaPredictor:
    """
    Direct vLLM LLM instance with parameters matching spatial_test_llava.py.

    Key difference from the old version:
    - Uses vLLM dict-based input: {"prompt": ..., "multi_modal_data": {"image": ...}}
    - No silent fallback to text-only generation on error.
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
        """
        Build prompt with proper chat template for vLLM.

        For llava-hf models the tokenizer has a chat_template that
        understands {"type": "image"} entries. We try that first and
        fall back to a simple USER/ASSISTANT wrapper.
        """
        options_str = "; ".join(options)
        # Prompt text matching the reference spatial_test_llava.py exactly.
        # The reference uses literal \\n (backslash-n) in the f-string.
        prompt_text = (
            f'You are currently a senior expert in spatial relation reasoning. \\n'
            f' Given an Image, a Question and Options, your task is to answer the '
            f'correct spatial relation. Note that you only need to choose one option '
            f'from the all options without explaining any reason. \\n'
            f' Input: Image: <image>, Question: {question}, Options: {options_str}. \\n'
            f' Output:'
        )

        # --- Strategy 1: Use HF chat template (works for llava-hf models) ---
        try:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            return prompt
        except Exception as e:
            logger.warning(f"Chat template with image dict failed: {e}")

        # --- Strategy 2: Text-only chat template + manual <image> token ---
        try:
            messages = [
                {"role": "user", "content": prompt_text}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            # Ensure <image> is present for vLLM to bind the pixel data
            if "<image>" not in prompt:
                prompt = "<image>\n" + prompt
            return prompt
        except Exception as e:
            logger.warning(f"Text chat template also failed: {e}")

        # --- Strategy 3: Manual LLaVA-v1 format ---
        logger.warning("Using manual LLaVA-v1 prompt format as last resort.")
        prompt = (
            f"USER: <image>\n{prompt_text}\n"
            f"ASSISTANT:"
        )
        return prompt

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: List[str], answer: str) -> str:
        prompt = self._build_prompt(question, options)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return "--"

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p if self.top_p is not None else 1.0,
            stop=[],
        )

        # ── Correct vLLM multimodal API ──────────────────────────────
        # Pass image via dict-based input, NOT as a keyword argument.
        try:
            outputs = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                },
                sampling_params=sampling_params,
            )
        except Exception as e:
            # Do NOT silently fall back to text-only — that hides the real bug.
            logger.error(f"vLLM generate failed: {e}")
            import traceback
            traceback.print_exc()
            return "--"

        raw_output = outputs[0].outputs[0].text.strip()
        return raw_output


def run_infer(args, config: ExperimentConfig):
    model_path = config.model.model_name_or_path
    model_type = config.model.model_type.lower()

    if not check_vllm_available():
        logger.error("vLLM not installed. Install with: pip install vllm")
        return

    logger.info(f"Loading vLLM LLM for {model_type} inference...")

    try:
        from vllm import LLM
    except ImportError as e:
        logger.error(f"vLLM import failed: {e}")
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
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

    # Debug: log first 3 prompts + raw outputs
    debug_printed = 0

    for index, item in enumerate(tqdm(test_data, desc="Inference", unit="img", ncols=100)):
        image_path = image_dir / item["image"]
        raw_output = predictor.predict(
            str(image_path),
            item["question"],
            item.get("options", []),
            item.get("answer", "")
        )

        # Debug: log first 3 raw outputs to verify model is actually generating
        if debug_printed < 3:
            logger.info(f"[DEBUG] Sample {index} raw output: '{raw_output}'")
            debug_printed += 1

        if not raw_output or len(raw_output.strip()) == 0:
            raw_output = "--"

        answer = item.get("answer", "")

        # Same matching logic as spatial_test_llava.py
        is_correct = 0
        output_lower = raw_output.lower()
        answer_lower = answer.lower()
        if output_lower in answer_lower or answer_lower in output_lower:
            is_correct = 1
            right_count += 1

        predictions.append(build_result_record(item, index, raw_output))

        # Log progress every 20 items
        if (index + 1) % 20 == 0 or index == 0:
            acc = right_count / (index + 1)
            logger.info(
                f"[{index + 1}/{total}] Output: '{raw_output}' | Answer: '{answer}' | "
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
