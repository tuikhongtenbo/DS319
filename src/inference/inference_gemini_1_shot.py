"""
Inference predictor for multimodal models via OpenAI-compatible API (1-shot).
Uses PIL.Image for image loading and base64 encoding.
"""

import base64
import io
import random
import time
from pathlib import Path
from typing import List

import PIL.Image as Image
from openai import OpenAI

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
    def __init__(self, model_name: str, api_key: str, image_dir: str, base_url: str = "https://api.tokenlab.sh/v1"):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.image_dir = Path(image_dir)

        # Pre-select a random example
        self.example = random.choice(ALL_EXAMPLES)
        self.example_image = self._encode_image(self.image_dir / self.example["image"])

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
                "\nGiven the following 1 example to learn the spatial relation reasoning task:"
            )
            prompt2 = f"\n{self.example['text']}"
            prompt3 = f"\nInput: Image: \nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{self.example_image}"}
                            },
                            {"type": "text", "text": prompt2},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            },
                            {"type": "text", "text": prompt3},
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
