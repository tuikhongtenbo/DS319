"""
Data collators for variable-length VLM batches.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


def _pad_1d(tensors: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    max_len = max(tensor.size(0) for tensor in tensors)
    padded = []
    for tensor in tensors:
        pad_size = max_len - tensor.size(0)
        if pad_size > 0:
            padding = torch.full((pad_size,), padding_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)
        padded.append(tensor)
    return torch.stack(padded)


@dataclass
class BlipCollator:
    """Pad BLIP VQA batches to a common sequence length."""

    pad_token_id: int = 0

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = _pad_1d([item["input_ids"] for item in batch], self.pad_token_id)
        attention_mask = _pad_1d([item["attention_mask"] for item in batch], 0)
        labels = _pad_1d([item["labels"] for item in batch], self.pad_token_id)
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


@dataclass
class Blip2Collator:
    """Pad BLIP-2 batches; labels use ignore_index=1 per original repo."""

    pad_token_id: int = 1
    label_pad_token_id: int = 1

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = _pad_1d([item["input_ids"] for item in batch], self.pad_token_id)
        attention_mask = _pad_1d([item["attention_mask"] for item in batch], 0)
        labels = _pad_1d([item["labels"] for item in batch], self.label_pad_token_id)
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


@dataclass
class InstructBlipCollator:
    """Pad InstructBLIP batches."""

    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = _pad_1d([item["input_ids"] for item in batch], self.pad_token_id)
        attention_mask = _pad_1d([item["attention_mask"] for item in batch], 0)
        labels = _pad_1d([item["labels"] for item in batch], self.label_pad_token_id)
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        collated: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
        if "qformer_input_ids" in batch[0]:
            collated["qformer_input_ids"] = _pad_1d(
                [item["qformer_input_ids"] for item in batch], self.pad_token_id
            )
        if "qformer_attention_mask" in batch[0]:
            collated["qformer_attention_mask"] = _pad_1d(
                [item["qformer_attention_mask"] for item in batch], 0
            )
        return collated
