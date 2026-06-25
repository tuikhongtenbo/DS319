"""
Inference predictor for GPT-4 (0-shot).
"""

import base64
import logging
from typing import List
import openai

logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class GPTZeroShotPredictor:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        base64_image = encode_image(image_path)
        
        system_prompt = (
            "You are currently a senior expert in spatial relation reasoning. "
            "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
            "Note that you only need to choose one option from the all options without explaining any reason."
        )
        
        user_prompt = f"Input: Question: {question}, Options: {'; '.join(options)}.\nOutput:"
        
        messages = [
            {"role": "system", "content": system_prompt},
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
