"""
Project-wide constants.
"""

from typing import Dict, List

# Answer space for SpatialMQA
SPATIAL_RELATIONS: List[str] = [
    "on/above",
    "below",
    "left of",
    "right of",
    "in front of",
    "behind",
]

# Supported open-source models for inference
SUPPORTED_OPEN_MODELS = [
    "blip2-opt-2.7b",
    "instructblip-3b",
    "mplug-owl-7b",
    "llava-v1.5-7b",
    "spacellava",
]

# Supported closed-source models for inference
SUPPORTED_CLOSED_MODELS = [
    "gpt-4o",
]

# Supported models for finetuning
SUPPORTED_FINETUNE_MODELS = [
    "blip-vqa-base",
    "blip2-opt-2.7b",
    "llava-v1.5-7b",
    "spacellava",
]

# Deprecated models kept for compatibility only
DEPRECATED_MODELS = [
    "idefics",
    "instructblip-finetune",
    "qwen-3.6",
    "gemini",
]

# Default hyperparameters aligned with thamkhao/SpatialMQA
MODEL_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "blip-vqa-base": {
        "learning_rate": 6e-7,
        "num_epochs": 30,
        "batch_size": 8,
        "patience": 5,
    },
    "blip2-opt-2.7b": {
        "learning_rate": 4e-5,
        "num_epochs": 30,
        "batch_size": 8,
        "patience": 5,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "llava-v1.5-7b": {
        "learning_rate": 2e-4,
        "num_epochs": 10,
        "batch_size": 8,
        "lora_r": 128,
        "lora_alpha": 256,
    },
    "spacellava": {
        "learning_rate": 2e-4,
        "num_epochs": 2,
        "batch_size": 8,
        "lora_r": 128,
        "lora_alpha": 256,
    },
}
