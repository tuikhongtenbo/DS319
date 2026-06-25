"""
Prompt builders and dataset path helpers for SpatialMQA.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

from ..constants import SPATIAL_RELATIONS


def resolve_split_paths(data_path: Union[str, Path]) -> Tuple[Path, Path]:
    """Resolve train and validation JSONL paths from a file or directory."""
    path = Path(data_path)
    if path.is_dir():
        train_path = path / "train.jsonl"
        val_path = path / "dev.jsonl"
        if not val_path.exists():
            val_path = train_path
    else:
        train_path = path
        val_path = path
    return train_path, val_path


def resolve_test_path(data_path: Union[str, Path]) -> Path:
    """Resolve test JSONL path from a file or directory."""
    path = Path(data_path)
    if path.is_dir():
        test_path = path / "test.jsonl"
        if not test_path.exists():
            test_path = path / "dev.jsonl"
        return test_path
    return path


def get_sample_id(item: Dict, index: int) -> int:
    """Return sample id from record or fall back to index."""
    sample_id = item.get("id", index)
    if isinstance(sample_id, str) and sample_id.isdigit():
        return int(sample_id)
    return int(sample_id) if isinstance(sample_id, int) else index


def build_spatial_prompt(question: str, options: List[str]) -> str:
    """Standard SpatialMQA prompt used by BLIP2, InstructBLIP, LLaVA, and mPLUG."""
    options_str = "; ".join(options)
    return (
        "You are currently a senior expert in spatial relation reasoning. \n"
        " Given an Image, a Question and Options, your task is to answer the "
        "correct spatial relation. Note that you only need to choose one option "
        "from the all options without explaining any reason. \n"
        f" Input: Image: <image>, Question: {question}, Options: {options_str}. \n"
        " Output:"
    )


def decode_blip_output(processor, generated_ids, prompt: str) -> str:
    """
    Decode BLIP-VQA generation output.

    BLIP returns decoder token ids; slicing by encoder ``input_ids`` length
    produces empty strings. The original SpatialMQA repo decodes the full
    sequence, then optionally strips an echoed prompt prefix.
    """
    full_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    if not full_text:
        return ""

    prompt_clean = prompt.strip()
    if full_text.lower().startswith(prompt_clean.lower()):
        return full_text[len(prompt_clean):].strip()
    return full_text


def build_blip_prompt(question: str, options: List[str]) -> str:
    """BLIP zero-shot / finetuned prompt format from the original repo."""
    if not options:
        return question
    if len(options) == 1:
        return f"{question} {options[0]}"
    joined = ", ".join(options[:-1])
    return f"{question} {joined} or {options[-1]}"


def build_spacellava_prompt(question: str, options: List[str]) -> str:
    """SpaceLLaVA zero-shot prompt from the original repo."""
    options_str = "; ".join(options)
    return f"Question: {question} \nOptions: {options_str} \nAnswer:"


def normalize_blip_answer(text: str) -> str:
    """Normalize BLIP finetuned output for on/above relation."""
    normalized = text.lower().strip()
    return normalized.replace("on / above", "on/above").replace("on/ above", "on/above")


def extract_predicted_relation(output: str, options: List[str] = None) -> str:
    """
    Map free-form model text to a single spatial relation.

    Uses longest-option-first matching to avoid partial hits such as
    matching ``in`` before ``in front of``.
    """
    text = output.lower().strip()
    if not text:
        return ""

    candidates = options if options else SPATIAL_RELATIONS
    unique_candidates = sorted({candidate.lower() for candidate in candidates}, key=len, reverse=True)
    for candidate in unique_candidates:
        if candidate in text:
            return candidate
    return text


def match_prediction(output: str, answer: str, options: List[str] = None) -> bool:
    """
    Check whether model output matches the ground-truth relation.

    Avoids the false-positive pattern ``output in answer`` which marks empty
    strings and fragments like ``of`` or ``left`` as correct.
    """
    if output is None or not str(output).strip():
        return False

    predicted = extract_predicted_relation(output, options)
    if not predicted:
        return False

    answer_lower = answer.lower().strip()
    if predicted == answer_lower:
        return True

    # Allow answer contained in a longer but still valid generation.
    return answer_lower in predicted


def build_result_record(item: Dict, index: int, output: str) -> Dict:
    """Build a prediction record compatible with metrics.py."""
    answer = item["answer"]
    options = item.get("options", [])
    normalized_output = extract_predicted_relation(output, options)
    record = {
        "id": get_sample_id(item, index),
        "result": 1 if match_prediction(output, answer, options) else 0,
        "output": normalized_output,
        "answer": answer,
    }
    if options:
        record["options"] = options
    return record
