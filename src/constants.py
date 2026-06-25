"""
Project-wide constants.
"""

from typing import List

# Answer space for SpatialMQA
SPATIAL_RELATIONS: List[str] = [
    "on/above",
    "below",
    "left of",
    "right of",
    "in front of",
    "behind"
]

# Supported open-source models for inference
SUPPORTED_OPEN_MODELS = [
    "blip2-opt-2.7b",
    "instructblip-3b",
    "mplug-owl-7b",
    "llava-v1.5-7b",
    "spacellava"
]

# Supported closed-source models for inference
SUPPORTED_CLOSED_MODELS = [
    "gpt-4o",
    "qwen-3.6"
]

# Supported models for finetuning
SUPPORTED_FINETUNE_MODELS = [
    "blip-vqa-base",
    "blip2-opt-2.7b",
    "llava-v1.5-7b",
    "spacellava"
]
