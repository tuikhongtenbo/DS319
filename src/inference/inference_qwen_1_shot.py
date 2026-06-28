"""
Inference predictor for Qwen via DashScope (1-shot).
"""

import base64
from pathlib import Path
from typing import List

from openai import OpenAI

from ..utils.io import load_jsonl
from .inference_qwen_0_shot import DASHSCOPE_BASE_URL, resolve_qwen_api_key


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class QwenOneShotPredictor:
    def __init__(self, model_name: str, api_key: str = "", train_data_path: str = "", image_dir: str = ""):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=resolve_qwen_api_key(api_key),
            base_url=DASHSCOPE_BASE_URL,
        )

        train_data = load_jsonl(train_data_path)
        if not train_data:
            raise ValueError("Training data is empty, cannot create 1-shot example.")

        sample = train_data[0]
        self.example_question = sample["question"]
        self.example_options = sample.get("options", [])
        self.example_answer = str(sample["answer"])
        sample_image_path = Path(image_dir) / sample["image"]
        self.example_base64_image = encode_image(str(sample_image_path))

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        base64_image = encode_image(image_path)

        system_prompt = (
            "You are currently a senior expert in spatial relation reasoning. "
            "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
            "Note that you only need to choose one option from all options without explaining any reason."
        )

        example_user_prompt = f"Input: Question: {self.example_question}, Options: {'; '.join(self.example_options)}.\nOutput:"
        user_prompt = f"Input: Question: {question}, Options: {'; '.join(options)}.\nOutput:"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example_user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.example_base64_image}"}},
                ],
            },
            {"role": "assistant", "content": self.example_answer},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=20,
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )

        return response.choices[0].message.content.strip()
