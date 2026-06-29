"""
Inference predictor for multimodal models via OpenAI-compatible API (0-shot).
Uses PIL.Image for image loading and base64 encoding.
"""

import base64
import io
import time
from pathlib import Path
from typing import List

import PIL.Image as Image
from openai import OpenAI


class GeminiZeroShotPredictor:
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.tokenlab.sh/v1"):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _encode_image(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        try:
            image_base64 = self._encode_image(image_path)

            prompt = (
                "You are currently a senior expert in spatial relation reasoning. "
                "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
                "Note that you only need to choose one option from all options without explaining any reason."
                f"\nInput: Image: \nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
            )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=20,
            )
            
            time.sleep(0.5)
            return response.choices[0].message.content.strip().rstrip(".").lower()
        
        except Exception as e:
            print(f"Error during inference: {e}")
            return ""
