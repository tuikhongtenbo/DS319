"""
Inference predictor for GPT-4 (1-shot).
Takes a sample from the training set dynamically.
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

class GPTOneShotPredictor:
    def __init__(self, model_name: str, api_key: str, train_data_path: str, image_dir: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
        
        # Load 1 sample from training set for the 1-shot example
        train_data = load_jsonl(train_data_path)
        if len(train_data) > 0:
            sample = train_data[0] # Take the first sample
            self.example_question = sample["question"]
            self.example_options = sample.get("options", [])
            self.example_answer = sample["answer"]
            
            sample_image_path = Path(image_dir) / sample["image"]
            self.example_base64_image = encode_image(str(sample_image_path))
        else:
            raise ValueError("Training data is empty, cannot create 1-shot example.")

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        base64_image = encode_image(image_path)
        
        system_prompt = (
            "You are currently a senior expert in spatial relation reasoning. "
            "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
            "Note that you only need to choose one option from the all options without explaining any reason."
        )
        
        # 1-shot Example User Prompt
        example_user_prompt = f"Input: Question: {self.example_question}, Options: {'; '.join(self.example_options)}.\nOutput:"
        
        # Target Query
        user_prompt = f"Input: Question: {question}, Options: {'; '.join(options)}.\nOutput:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            # Few-shot user example with image
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example_user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.example_base64_image}"
                        }
                    }
                ]
            },
            # Few-shot assistant answer
            {
                "role": "assistant",
                "content": str(self.example_answer)
            },
            # Current query
            {
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
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=20,
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
