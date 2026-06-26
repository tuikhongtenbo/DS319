"""
vLLM-accelerated inference for LLaVA and SpaceLLaVA.

Requires: pip install vllm>=0.4.0
Supported: llava-hf/llava-1.5, llava-1.6, SpaceLLaVA
Qwen2-VL: use inference_vllm_qwen.py instead.
"""

import re
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from ..configs.config import ExperimentConfig
from ..datasets.preprocessing import build_result_record, build_spatial_prompt, resolve_test_path
from ..metrics.metrics import calculate_spatial_metrics
from ..utils.io import load_jsonl, save_jsonl
from ..utils.logging import setup_logger
from .inference_vllm_base import check_vllm_available, build_sampling_params

logger = setup_logger(__name__)

IMAGE_PLACEHOLDER = "<image>"


def _conv_prompt(question: str, options: List[str], conv_mode: str) -> str:
    prompt_text = build_spatial_prompt(question, options)

    # Minimal prompt formatting that does not require the `llava` package.
    # vLLM will apply the model's chat template during generation, so we only
    # need to ensure the image placeholder is present.
    image_token = "<image>"
    if IMAGE_PLACEHOLDER in prompt_text:
        prompt_text = re.sub(IMAGE_PLACEHOLDER, image_token, prompt_text)
    else:
        prompt_text = image_token + "\n" + prompt_text
    return prompt_text


def _detect_conv_mode(model_name_or_path: str) -> str:
    m = model_name_or_path.lower()
    if "llama-2" in m:
        return "llava_llama_2"
    if "mistral" in m or "mixtral" in m:
        return "mistral_instruct"
    if "v1" in m:
        return "llava_v1"
    return "llava_v0"


class VLLMLlavaPredictor:
    def __init__(self, llm, conv_mode: str, max_new_tokens: int = 20):
        self.llm = llm
        self.conv_mode = conv_mode
        self.sampling_params = build_sampling_params(max_new_tokens, temperature=0.0)

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        prompt = _conv_prompt(question, options, self.conv_mode)
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
            outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip().rstrip(".")


def run_infer(args, config: ExperimentConfig):
    if not check_vllm_available():
        logger.error("vLLM not installed. Install with: pip install vllm")
        return

    model_path = config.model.model_name_or_path
    model_type = config.model.model_type.lower()

    logger.info(f"Loading vLLM for {model_type} inference...")

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

    predictor = VLLMLlavaPredictor(llm=llm, conv_mode=conv_mode, max_new_tokens=20)

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    logger.info("Starting vLLM LLaVA inference...")
    for index, item in enumerate(tqdm(test_data)):
        image_path = image_dir / item["image"]
        output = predictor.predict(str(image_path), item["question"], item.get("options", []))
        if not output:
            output = "--"
        predictions.append(build_result_record(item, index, output))

    out_results = Path(args.out_results) if args.out_results else Path("results")
    out_results.mkdir(parents=True, exist_ok=True)
    out_path = out_results / "predictions.jsonl"
    save_jsonl(predictions, str(out_path))

    metrics = calculate_spatial_metrics(predictions)
    logger.info("--- vLLM LLaVA Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
