"""
Inference predictor for Gemini 3.1 Flash-Lite (0-shot).
Uses PIL.Image for image loading and minimal thinking mode.
"""

import re
import time
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

    def _parse_retry_delay(self, error_message: str) -> float:
        """Extract retry delay from Gemini API error message."""
        match = re.search(r'Please retry in ([\d.]+)s', error_message)
        if match:
            return float(match.group(1))
        return 60.0  # Default to 60 seconds

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                # Load image using PIL (not base64 in message)
                image = Image.open(image_path)

                prompt = (
                    "You are currently a senior expert in spatial relation reasoning. "
                    "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
                    "Note that you only need to choose one option from all options without explaining any reason."
                    f"\nInput: Image: \nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
                )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image],
                    config=self.generation_config,
                )
                return response.text.strip().rstrip(".").lower()
            
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    retry_delay = self._parse_retry_delay(error_str)
                    print(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting {retry_delay:.1f}s...")
                    time.sleep(retry_delay + 1)  # Add 1s buffer
                elif attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Error during inference (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error during inference after {max_retries} attempts: {e}")
                    return ""
        
        return ""
