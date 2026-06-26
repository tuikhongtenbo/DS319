"""
vLLM inference engine utilities.

Provides:
  - check_vllm_available()   — safe import guard
  - build_sampling_params()  — vLLM SamplingParams factory
  - load_vllm_engine()       — guarded engine loader (used by inference scripts)

vLLM is optional. All functions return None / False gracefully if vLLM
is not installed, so the caller can fall back to HuggingFace inference
without crashing.
"""

from ..utils.logging import setup_logger

logger = setup_logger(__name__)

VLLM_SUPPORTED_ARCHS = {
    "llava",
    "qwen2_vl",
}


def check_vllm_available() -> bool:
    """Return True only if vLLM is importable."""
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
    """Build vLLM SamplingParams with defaults suitable for short VQA answers."""
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("vLLM is not installed. Run: pip install vllm")

    return SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop=[],
        include_stop_str_in_output=False,
    )


def is_vllm_supported(model_type: str, model_name_or_path: str) -> bool:
    """Return True if the model architecture is known to be supported by vLLM."""
    combined = (model_type + " " + model_name_or_path).lower()
    return any(arch in combined for arch in VLLM_SUPPORTED_ARCHS)


def load_vllm_engine(
    model_name_or_path: str,
    model_type: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    trust_remote_code: bool = True,
):
    """
    Load a vLLM LLM engine.

    Returns None (with a warning log) if:
      - vLLM is not installed
      - the model architecture is not in the supported list
      - the engine fails to load
    Callers must handle None and fall back to HuggingFace.
    """
    if not check_vllm_available():
        logger.warning(
            "vLLM is not installed — falling back to HuggingFace inference. "
            "Install with: pip install vllm"
        )
        return None

    if not is_vllm_supported(model_type, model_name_or_path):
        logger.warning(
            f"Model '{model_name_or_path}' (type={model_type}) does not appear to be "
            f"a vLLM-supported architecture (supported: {VLLM_SUPPORTED_ARCHS}). "
            f"Falling back to HuggingFace inference."
        )
        return None

    ver = get_vllm_version()
    logger.info(
        f"Loading vLLM {ver} engine for {model_name_or_path} "
        f"(tp={tensor_parallel_size}, gpu_mem={gpu_memory_utilization}, "
        f"max_len={max_model_len})..."
    )

    try:
        from vllm import LLM
        engine = LLM(
            model=model_name_or_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        logger.info("vLLM engine loaded successfully.")
        return engine
    except Exception as e:
        logger.warning(f"vLLM engine failed to load ({e}) — falling back to HuggingFace.")
        return None
