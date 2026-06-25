"""
Inference predictors for Open-source and API-based models.
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def encode_image(image_path: str) -> str:
    """Encodes an image to base64 for API requests."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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
        
        # Format similar to original prompt
        prompt = (f"You are currently a senior expert in spatial relation reasoning.\n"
                  f"Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
                  f"Note that you only need to choose one option from the all options without explaining any reason.\n"
                  f"Input: Image: <image>, Question: {question}, Options: {'; '.join(options)}.\nOutput:")
                  
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        # Generation
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        
        # Decode
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        # Some processors include the prompt in the output, so we need to extract the answer
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
            
        return response.rstrip('.')


class APIPredictor:
    """Predictor for GPT-4o and Qwen-3.6 via API."""
    def __init__(self, model_name: str, api_key: str, shots: int = 0):
        self.model_name = model_name
        self.api_key = api_key
        self.shots = shots
        
        # For actual implementation, instantiate openai or dashscope client
        if "gpt-4" in model_name.lower():
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        elif "qwen" in model_name.lower():
            # Using openai API format which DashScope/Qwen supports or dashscope SDK
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key, 
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            raise ValueError(f"Unsupported API model: {model_name}")

    def get_few_shot_messages(self) -> List[Dict[str, Any]]:
        """Returns few-shot examples if shots > 0."""
        messages = []
        if self.shots >= 1:
            # Example 1-shot mapping (placeholders, would load from actual dataset ideally)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Input: Question: Where is the cat? Options: on the table; below the table. Output:"}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": "on the table"
            })
        return messages

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        base64_image = encode_image(image_path)
        
        system_prompt = (
            "You are currently a senior expert in spatial relation reasoning. "
            "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
            "Note that you only need to choose one option from the all options without explaining any reason."
        )
        
        user_prompt = f"Input: Question: {question}, Options: {'; '.join(options)}.\nOutput:"
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add few-shot examples
        messages.extend(self.get_few_shot_messages())
        
        # Add current query
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=20,
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
