"""
Inference predictor for Qwen models.
"""

import base64
import logging
from pathlib import Path
from typing import List
import openai
from ..utils.io import load_jsonl

logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class QwenPredictor:
    def __init__(self, model_name: str, api_key: str, shots: int = 0, train_data_path: str = None, image_dir: str = None):
        self.model_name = model_name
        self.shots = shots
        self.client = openai.OpenAI(
            api_key=api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.example_data = None
        if shots >= 1 and train_data_path and image_dir:
            train_data = load_jsonl(train_data_path)
            if len(train_data) > 0:
                sample = train_data[0]
                sample_image_path = Path(image_dir) / sample["image"]
                self.example_data = {
                    "question": sample["question"],
                    "options": sample.get("options", []),
                    "answer": sample["answer"],
                    "base64_image": encode_image(str(sample_image_path))
                }

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
        if self.shots >= 1 and self.example_data:
            example_prompt = f"Input: Question: {self.example_data['question']}, Options: {'; '.join(self.example_data['options'])}.\nOutput:"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": example_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.example_data['base64_image']}"}}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": str(self.example_data['answer'])
            })
            
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
