"""
Dataset classes and utilities for SpatialMQA.
"""

from pathlib import Path
from typing import Any, Dict, List
from PIL import Image

import torch
from torch.utils.data import Dataset
from ..utils.io import load_json, load_jsonl


class SpatialMQADataset(Dataset):
    """
    Unified dataset class for training HF-based models like BLIP and BLIP2.
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        processor: Any,
        max_samples: int = None,
        is_training: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.is_training = is_training
        
        path = Path(data_path)
        if path.suffix == ".jsonl":
            self.data = load_jsonl(path)
        else:
            self.data = load_json(path)
            
        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        question = item.get("question", "")
        answer = str(item.get("answer", ""))
        image_id = item.get("image", "")
        image_path = self.image_dir / image_id
        
        image = Image.open(image_path).convert("RGB")

        # Prepare inputs for processor
        encoding = self.processor(images=image, text=question, return_tensors="pt")

        if self.is_training:
            # Tokenize answer for labels
            labels = self.processor.tokenizer(
                answer, 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids
            
            # Append EOS token. Depending on the model, eos_token_id might be different.
            eos_token_id = self.processor.tokenizer.eos_token_id
            if eos_token_id is not None:
                labels = torch.cat((labels, torch.tensor([[eos_token_id]])), dim=1)
            
            encoding["labels"] = labels

        # Remove batch dimension added by processor
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)

        return encoding


def prepare_llava_dataset(data_path: str, output_path: str) -> None:
    """
    Converts standard SpatialMQA data into LLaVA conversational format.
    Useful for generating the required json for LLaVA/SpaceLLaVA training.
    """
    path = Path(data_path)
    if path.suffix == ".jsonl":
        data = load_jsonl(path)
    else:
        data = load_json(path)
        
    llava_data = []
    for idx, item in enumerate(data):
        llava_item = {
            "id": item.get("id", str(idx)),
            "image": item["image"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{item['question']}"
                },
                {
                    "from": "gpt",
                    "value": str(item["answer"])
                }
            ]
        }
        llava_data.append(llava_item)
        
    from ..utils.io import save_json
    save_json(llava_data, output_path)
