"""
Inference predictor for Gemini 3.1 Flash-Lite (1-shot).
Uses PIL.Image for image loading and minimal thinking mode.
"""

import random
import re
import time
from pathlib import Path
from typing import List

import PIL.Image as Image
from google import genai

from ..utils.io import load_jsonl


# Example prompts for each spatial relation type (for 1-shot)
EXAMPLE_LIST_RULE1 = [
    {
        "text": "Question: For the clock in the picture, which side of the 1 scale does the hour hand point to?, Options: left of; right of. Output: right of.",
        "image": "000000358641.jpg",
    },
    {
        "text": "Question: Where is the white plate located relative to the glass?, Options: in front of; behind; left of; right of. Output: in front of.",
        "image": "000000209618.jpg",
    },
    {
        "text": "Question: For the letters on the warning sign, where is the letter W located relative to the letter O?, Options: on/above; below; left of; right of. Output: below.",
        "image": "000000010682.jpg",
    },
]

EXAMPLE_LIST_RULE2 = [
    {
        "text": "Question: If you are the person skiing in the picture, where is your shadow located relative to you?, Options: in front of; behind; left of; right of. Output: right of.",
        "image": "000000057664.jpg",
    },
    {
        "text": "Question: If you were a player playing on the court, where would the tennis ball be located relative to you?, Options: on/above; below; in front of; behind; left of; right of. Output: on/above.",
        "image": "000000073924.jpg",
    },
    {
        "text": "Question: If you were the little girl in the picture, where would the window be located relative to you?, Options: in front of; behind; left of; right of. Output: behind.",
        "image": "000000022707.jpg",
    },
]

EXAMPLE_LIST_RULE3 = [
    {
        "text": "Question: If you are the driver of the bus in the picture, from your perspective, where is the stroller located relative to the bus?, Options: in front of; behind; left of; right of. Output: left of.",
        "image": "000000139664.jpg",
    },
    {
        "text": "If you are sitting in front of the computer in the picture, where is the scissors located relative to the laptop from your perspective?, Options: on/above; below; in front of; behind; left of; right of. Output: right of.",
        "image": "000000221101.jpg",
    },
    {
        "text": "Question: If you are the driver of the white car in the picture, from your perspective, where is the motorcycle located relative to the car?, Options: in front of; behind; left of; right of. Output: behind.",
        "image": "000000164692.jpg",
    },
]

ALL_EXAMPLES = EXAMPLE_LIST_RULE1 + EXAMPLE_LIST_RULE2 + EXAMPLE_LIST_RULE3


class GeminiOneShotPredictor:
    def __init__(self, model_name: str, api_key: str, image_dir: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.image_dir = Path(image_dir)

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

        # Pre-select a random example
        self.example = random.choice(ALL_EXAMPLES)
        self.example_image = Image.open(self.image_dir / self.example["image"])

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
                    "\nGiven the following 1 example to learn the spatial relation reasoning task:"
                )
                prompt2 = f"\n{self.example['text']}"
                prompt3 = f"\nInput: Image: \nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, self.example_image, prompt2, image, prompt3],
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
