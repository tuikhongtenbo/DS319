"""
vLLM-accelerated inference for Qwen2-VL.

Requires:  pip install vllm>=0.5.0
           (Qwen2-VL support landed in vLLM 0.5.x)

vLLM supports Qwen2-VL natively via its multimodal pipeline.
Only Qwen2-VL models are routed here.
"""

import base64
from pathlib import Path
from typing import List

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

VLLM_MIN_VERSION_QWEN = "0.5.0"

SYSTEM_PROMPT = (
    "You are currently a senior expert in spatial relation reasoning. "
    "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
    "Note that you only need to choose one option from all options without explaining any reason."
)


def _is_qwen2vl(model_type: str, model_name_or_path: str) -> bool:
    combined = (model_type + " " + model_name_or_path).lower()
    return "qwen" in combined and "vl" in combined


def _get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "0.0")
    except Exception:
        return "0.0"


def _check_vllm_ready() -> bool:
    if not check_vllm_available():
        logger.error("vLLM is not installed. Install it with: pip install vllm")
        return False
    ver = _get_vllm_version()
    import packaging.version
    if packaging.version.parse(ver) < packaging.version.parse(VLLM_MIN_VERSION_QWEN):
        logger.error(
            f"vLLM {ver} detected but {VLLM_MIN_VERSION_QWEN}+ is required for Qwen2-VL. "
            f"Upgrade: pip install 'vllm>={VLLM_MIN_VERSION_QWEN}'"
        )
        return False
    return True


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_messages(question: str, options: List[str], image_path: str) -> List[dict]:
    opts_str = "; ".join(options) if options else ""
    image_b64 = _encode_image(image_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": f"Question: {question}\nOptions: {opts_str}\nOutput:"},
            ],
        },
    ]


class VLLMQwenPredictor:
    def __init__(self, llm, max_new_tokens: int = 20, temperature: float = 0.0):
        self.llm = llm
        self.sampling_params = build_sampling_params(max_new_tokens, temperature)

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        messages = _build_messages(question, options, image_path)
        try:
            # vLLM 0.5+: use chat()
            outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
        except AttributeError:
            # vLLM 0.4 fallback (should not happen since we require 0.5+)
            outputs = self.llm.generate(
                [messages[1]["content"][1]["text"]],
                self.sampling_params,
            )
        return outputs[0].outputs[0].text.strip().rstrip(".")


def run_infer(args, config: ExperimentConfig):
    model_type = config.model.model_type.lower()
    model_path = config.model.model_name_or_path

    if not _check_vllm_ready():
        return

    if not _is_qwen2vl(model_type, model_path):
        logger.warning(
            f"Only Qwen2-VL models are supported here, got '{model_path}' (type={model_type}). "
            f"Skipping."
        )
        return

    ver = _get_vllm_version()
    logger.info(
        f"Loading vLLM {ver} for Qwen2-VL "
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

    predictor = VLLMQwenPredictor(llm=llm, max_new_tokens=20, temperature=0.0)

    target_path = resolve_test_path(args.jsonl_dir or config.dataset.data_path)
    logger.info(f"Loading test dataset from {target_path}")
    test_data = load_jsonl(str(target_path))
    image_dir = Path(args.image_dir or config.dataset.image_dir)

    predictions = []
    logger.info("Starting vLLM Qwen2-VL inference...")
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
    logger.info("--- vLLM Qwen2-VL Evaluation Results ---")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"Accuracy X:  {metrics['accuracy_x']:.4f}")
    logger.info(f"Accuracy Y:  {metrics['accuracy_y']:.4f}")
    logger.info(f"Accuracy Z:  {metrics['accuracy_z']:.4f}")
