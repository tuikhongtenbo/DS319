"""
Inference predictor for Gemini 3.1 Flash-Lite (0-shot).
Uses PIL.Image for image loading and minimal thinking mode.
"""

from pathlib import Path
from typing import List

import PIL.Image as Image
from google import genai


class GeminiZeroShotPredictor:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

        self.generation_config = {
            "temperature": 1.0,  # Gemini 3 default
            "max_output_tokens": 20,
        }
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        # Load image using PIL (not base64 in message)
        image = Image.open(image_path)

        prompt = (
            "You are currently a senior expert in spatial relation reasoning. "
            "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
            "Note that you only need to choose one option from all options without explaining any reason."
            f"\nInput: Image: \nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=self.generation_config,
            )
            return response.text.strip().rstrip(".").lower()
        except Exception as e:
            print(f"Error during inference: {e}")
            return ""
