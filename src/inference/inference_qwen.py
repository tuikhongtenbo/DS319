"""
API-based inference for Qwen-VL via DashScope.
Supports 0-shot and 1-shot modes.
"""

import base64
from pathlib import Path
from typing import List

import openai

from ..utils.io import load_jsonl


SYSTEM_PROMPT = (
    "You are currently a senior expert in spatial relation reasoning. "
    "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
    "Note that you only need to choose one option from all options without explaining any reason."
)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class QwenZeroShotPredictor:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        base64_image = _encode_image(image_path)
        opts_str = "; ".join(options) if options else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {question}\nOptions: {opts_str}\nOutput:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=20,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


class QwenOneShotPredictor:
    def __init__(self, model_name: str, api_key: str, train_data_path: str, image_dir: str):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        train_data = load_jsonl(train_data_path)
        if not train_data:
            raise ValueError("Training data is empty, cannot create 1-shot example.")

        sample = train_data[0]
        self.example_question = sample["question"]
        self.example_options = sample.get("options", [])
        self.example_answer = str(sample["answer"])
        sample_image_path = Path(image_dir) / sample["image"]
        self.example_base64_image = _encode_image(str(sample_image_path))

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        base64_image = _encode_image(image_path)
        opts_str = "; ".join(options) if options else ""

        example_user_prompt = (
            f"Question: {self.example_question}\n"
            f"Options: {'; '.join(self.example_options)}\nOutput:"
        )
        user_prompt = f"Question: {question}\nOptions: {opts_str}\nOutput:"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example_user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{self.example_base64_image}"}},
                ],
            },
            {"role": "assistant", "content": self.example_answer},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=20,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
