"""
vLLM inference engine utilities.

All functions are no-ops (return None/False) if vLLM is not installed,
allowing callers to fall back to HuggingFace inference without crashing.
"""

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def check_vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "0.0")
    except Exception:
        return "0.0"


def build_sampling_params(max_new_tokens: int = 20, temperature: float = 0.0):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop=[],
        include_stop_str_in_output=False,
    )


def is_vllm_supported(model_type: str, model_name_or_path: str) -> bool:
    supported = {"llava", "spacellava", "qwen_vl"}
    combined = (model_type + " " + model_name_or_path).lower()
    return any(s in combined for s in supported)


def load_vllm_engine(
    model_name_or_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
):
    if not check_vllm_available():
        logger.warning("vLLM not installed — falling back to HuggingFace.")
        return None

    if not is_vllm_supported("", model_name_or_path):
        logger.warning(f"vLLM does not support '{model_name_or_path}'. Falling back.")
        return None

    ver = get_vllm_version()
    logger.info(f"Loading vLLM {ver} engine for {model_name_or_path}...")

    try:
        from vllm import LLM
        return LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    except Exception as e:
        logger.warning(f"vLLM engine failed ({e}) — falling back to HuggingFace.")
        return None
