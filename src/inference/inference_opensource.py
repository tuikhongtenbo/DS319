"""
Inference predictor for open-source HuggingFace models.
"""

import logging
from typing import List, Any
import torch
from PIL import Image

logger = logging.getLogger(__name__)

class OpenSourcePredictor:
    """Predictor for local HuggingFace models."""
    def __init__(self, model: Any, processor: Any, device: torch.device):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        image = Image.open(image_path).convert("RGB")
        
        prompt = (f"You are currently a senior expert in spatial relation reasoning.\n"
                  f"Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
                  f"Note that you only need to choose one option from the all options without explaining any reason.\n"
                  f"Input: Image: <image>, Question: {question}, Options: {'; '.join(options)}.\nOutput:")
                  
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Determine the correct dtype from the model's parameters
        model_dtype = next(self.model.parameters()).dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
            
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
            
        return response.rstrip('.')
