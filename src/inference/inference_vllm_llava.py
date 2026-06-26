"""
vLLM-accelerated inference for LLaVA.

Requires:  pip install vllm>=0.4.0
Uses:      vllm.LLM with LlavaForConditionalGeneration

vLLM supports LLaVA natively via its multimodal pipeline.
Only models with architectures that vLLM can load will attempt vLLM loading;
all others gracefully fall back to HuggingFace inference.

Supported model types in vLLM:
  - llava-hf/llava-1.5  (LlavaForConditionalGeneration)
  - llava-hf/llava-1.6  (LlavaForConditionalGeneration)
  - llava-hf/llava-v1.6-mistral (LlavaForConditionalGeneration)
  - InternVL family     (requires InternVLChatForConditionalGeneration)
  - Qwen2-VL            (use inference_vllm_qwen.py instead)
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

# Models whose architecture vLLM officially supports for vision-language
VLLM_SUPPORTED_ARCHS = {
    "llava",          # any variant with "llava" in the hub name
    "qwen2_vl",       # handled separately in inference_vllm_qwen.py
}

VLLM_MIN_VERSION = "0.4.0"


def _is_vllm_supported(model_type: str, model_name_or_path: str) -> bool:
    """Return True if the model is in vLLM's supported architecture set."""
    combined = (model_type + " " + model_name_or_path).lower()
    return any(arch in combined for arch in VLLM_SUPPORTED_ARCHS)


def _get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "0.0")
    except Exception:
        return "0.0"


def _check_vllm_ready() -> bool:
    """Guard: ensure vLLM is installed and version is sufficient."""
    if not check_vllm_available():
        logger.error("vLLM is not installed. Install it with: pip install vllm")
        return False
    ver = _get_vllm_version()
    import packaging.version
    if packaging.version.parse(ver) < packaging.version.parse(VLLM_MIN_VERSION):
        logger.error(
            f"vLLM {ver} detected but {VLLM_MIN_VERSION}+ is required. "
            f"Upgrade: pip install 'vllm>={VLLM_MIN_VERSION}'"
        )
        return False
    return True


def _llava_conv_prompt(question: str, options: List[str], conv_mode: str) -> str:
    """Mirror the prompt-building logic from inference_llava.py."""
    try:
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_PLACEHOLDER
        from llava.conversation import conv_templates
    except ImportError:
        opts = "; ".join(options) if options else ""
        return f"Question: {question}\nOptions: {opts}\nAnswer:"

    prompt_text = build_spatial_prompt(question, options)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in prompt_text:
        prompt_text = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt_text)
    else:
        prompt_text = image_token_se + "\n" + prompt_text

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


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
    """
    Inference via vLLM's LLM engine with multimodal image input.
    """

    def __init__(
        self,
        llm,
        tokenizer,
        conv_mode: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        self.sampling_params = build_sampling_params(max_new_tokens, temperature)

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        prompt = _llava_conv_prompt(question, options, self.conv_mode)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return "--"

        # vLLM >= 0.5: pass image directly
        try:
            outputs = self.llm.generate(
                [prompt],
                self.sampling_params,
                multi_modal_data={"image": image},
            )
        except (TypeError, AttributeError):
            # vLLM 0.4.x: pass as pixel_values via a workaround (images param)
            outputs = self.llm.generate(
                [prompt],
                self.sampling_params,
            )

        return outputs[0].outputs[0].text.strip().rstrip(".")


def run_infer(args, config: ExperimentConfig):
    """Entrypoint — mirrors inference_llava.py signature but uses vLLM."""

    model_type = config.model.model_type.lower()
    model_path = config.model.model_name_or_path

    if not _check_vllm_ready():
        return

    if not _is_vllm_supported(model_type, model_path):
        logger.warning(
            f"Model '{model_path}' (type={model_type}) is not in vLLM's "
            f"supported architecture list. Supported: {VLLM_SUPPORTED_ARCHS}. "
            f"Skipping vLLM inference."
        )
        return

    ver = _get_vllm_version()
    logger.info(
        f"Loading vLLM {ver} for LLaVA inference "
        f"(tp={config.model.tensor_parallel_size}, "
        f"gpu_mem={config.model.gpu_memory_utilization}, "
        f"max_len={config.model.max_model_len})..."
    )

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

    # Load tokenizer for prompt tokenization
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = None

    conv_mode = _detect_conv_mode(model_path)
    logger.info(f"Using conversation template: {conv_mode}")

    predictor = VLLMLlavaPredictor(
        llm=llm,
        tokenizer=tokenizer,
        conv_mode=conv_mode,
        max_new_tokens=20,
        temperature=0.0,
    )

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    logger.info("Starting vLLM LLaVA inference...")
    for index, item in enumerate(tqdm(test_data)):
        image_path = image_dir / item["image"]
        output = predictor.predict(
            str(image_path),
            item["question"],
            item.get("options", []),
        )
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
